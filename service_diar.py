import os
import json
import subprocess
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import whisperx
from nemo.collections.asr.models.msdd_models import ClusteringDiarizer
from omegaconf import OmegaConf
import wget
from deepmultilingualpunctuation import PunctuationModel
import re

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_wav(input_file, base_file_name):
    output_file = os.path.join(os.path.dirname(input_file), f'{base_file_name}.wav')
    command = [
        'ffmpeg',
        '-i', input_file,
        '-acodec', 'pcm_s16le',
        '-ac', '1',
        '-ar', '16000',
        output_file
    ]
    logging.info(f'Выполнение команды конвертации: {" ".join(command)}')
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info(f'Результат конвертации: {result.stdout.decode()}')
    logging.info(f'ЛОГ: {result.stderr.decode()}')
    
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        logging.info(f'Файл {output_file} создан. Размер: {file_size} байт')
    else:
        logging.error(f'Файл {output_file} не создан после выполнения ffmpeg')
        raise FileNotFoundError(f'Файл {output_file} не создан после выполнения ffmpeg')
    
    return output_file

def transcribe_audio(input_audio_file, output_file):
    try:
        logging.info(f"Загрузка и предобработка файла: {input_audio_file}")
        if not os.path.exists(input_audio_file):
            logging.error(f"Файл {input_audio_file} не существует")
            raise FileNotFoundError(f"Файл {input_audio_file} не найден")
        
        file_size = os.path.getsize(input_audio_file)
        logging.info(f"Размер файла {input_audio_file}: {file_size} байт")
        
        audio = input_audio_file

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Используется: {device}")

        logging.info("Загрузка Whisper модели...")
        model = whisperx.load_model("large-v3", device, compute_type="int8", asr_options={"hotwords": []})

        logging.info("Транскрибирование аудио...")
        result = model.transcribe(audio, language="ru", batch_size=32)
        logging.info(f"Транскрибирование успешно. Определенный язык разговора: {result['language']}")

        logging.info("Обработка транскрипции...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=True
        )

        logging.info("Постобработка результатов...")
        for segment in result.get('segments', []):
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            words = segment.get('words', [])
            if words:
                segment['words'] = fill_missing_timestamps(words, segment_start, segment_end)

        logging.info("Сохранение результатов в формате JSON...")
        output_data = {"segments": []}
        for segment in result.get("segments", []):
            segment_data = {
                "start": round(segment.get('start', 0), 3),
                "end": round(segment.get('end', 0), 3),
                "text": segment.get('text', ''),
                "words": []
            }
            for word in segment.get("words", []):
                word_data = {
                    "word": word.get('word', ''),
                    "start": round(word.get('start', 0), 3),
                    "end": round(word.get('end', 0), 3),
                    "score": round(word.get('score', 0), 3)
                }
                segment_data["words"].append(word_data)
            output_data["segments"].append(segment_data)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Транскрипция завершена и сохранена в {output_file}")

        del model
        del model_a
        torch.cuda.empty_cache()

        return output_data

    except Exception as e:
        logging.error(f"Произошла ошибка: {str(e)}", exc_info=True)
        raise

def fill_missing_timestamps(words, segment_start, segment_end):
    known_words = [word for word in words if word.get('start') is not None and word.get('start') > 0]
    unknown_words = [word for word in words if word.get('start') is None or word.get('start') == 0]

    if not unknown_words:
        return words

    prev_known_word = None
    for word in reversed(known_words):
        if word['start'] < unknown_words[0].get('start', float('inf')):
            prev_known_word = word
            break

    if not prev_known_word:
        prev_known_word = {"word": "", "start": segment_start, "end": segment_start}

    next_known_word = None
    for word in known_words:
        if word['start'] > unknown_words[-1].get('start', float('-inf')):
            next_known_word = word
            break

    if not next_known_word:
        next_known_word = {"word": "", "start": segment_end, "end": segment_end}

    total_unknown_chars = sum(len(word['word']) for word in unknown_words)
    available_duration = max(next_known_word['start'] - prev_known_word['end'], 0)

    current_start = prev_known_word['end']

    for word in unknown_words:
        word_duration = (len(word['word']) / total_unknown_chars) * available_duration
        word['start'] = current_start
        word['end'] = current_start + word_duration
        word['score'] = 0.0
        current_start = word['end']

    for word in words:
        word['start'] = round(word['start'], 3)
        word['end'] = round(word['end'], 3)

    return words

def apply_punctuation(wsm):
    logger.info("Применение пунктуации...")
    punct_model = PunctuationModel()
    words_list = list(map(lambda x: x['word'], wsm))
    
    try:
        labeled_words = punct_model.predict(words_list)
    except Exception as e:
        logger.error(f"Ошибка при предсказании пунктуации: {e}")
        return wsm

    ending_puncts = '.?!'
    model_puncts = '.,;:!?'
    is_acronym = lambda x: re.fullmatch(r"\b(?:[а-яА-я]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labeled_words):
        word = word_dict['word']
        if word and labeled_tuple[1] in ending_puncts and (word[-1] not in model_puncts or is_acronym(word)):
            word += labeled_tuple[1]
            if word.endswith('..'): 
                word = word.rstrip('.')
            word_dict['word'] = word

    logger.info("Пунктуация применена успешно")
    return wsm

def download_model_config():
    MODEL_CONFIG = os.path.join('./', 'diar_infer_meeting.yaml')
    if not os.path.exists(MODEL_CONFIG):
        logger.info("Конфигурация модели не найдена локально. Начинаем загрузку...")
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml"
        try:
            MODEL_CONFIG = wget.download(config_url, './')
            logger.info(f"Конфигурация модели загружена и сохранена локально: {MODEL_CONFIG}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации модели: {e}")
            raise
    else:
        logger.info(f"Используется локальная конфигурация модели: {MODEL_CONFIG}")
    return MODEL_CONFIG

def configure_diarizer(config_path, manifest_path):
    logger.info("Настройка диаризатора...")
    try:
        config = OmegaConf.load(config_path)
        config.num_workers = 4
        config.batch_size = 32
        config.diarizer.manifest_filepath = manifest_path
        config.diarizer.out_dir = os.path.join('./', 'diarized')
        config.diarizer.speaker_embeddings.model_path = 'titanet_large'
        config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5, 1.0, 0.5]
        config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75, 0.5, 0.25]
        config.diarizer.speaker_embeddings.parameters.multiscale_weights = [0.33, 0.33, 0.33]
        config.diarizer.speaker_embeddings.parameters.save_embeddings = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        config.diarizer.clustering.parameters.max_num_speakers = 8
        config.diarizer.clustering.parameters.min_samples = 2
        config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic'
        config.diarizer.oracle_vad = False
        config.diarizer.vad.model_path = 'vad_multilingual_marblenet'
        config.diarizer.vad.parameters.onset = 0.8
        config.diarizer.vad.parameters.offset = 0.6
        config.diarizer.vad.parameters.pad_offset = -0.05
        logger.info("Диаризатор настроен успешно")
        return config
    except Exception as e:
        logger.error(f"Ошибка при настройке диаризатора: {e}")
        raise

def perform_diarization(config):
    logger.info("Начало процесса диаризации...")
    try:
        model = ClusteringDiarizer(cfg=config)
        logger.info("Модель диаризации инициализирована.")
        model.diarize()
        logger.info("Диаризация завершена успешно")
    except Exception as e:
        logger.error(f"Ошибка при выполнении диаризации: {e}")
        raise

def read_rttm_file(audio_file):
    rttm_file = f'./diarized/pred_rttms/{os.path.splitext(os.path.basename(audio_file))[0]}.rttm'
    logger.info(f"Чтение RTTM файла: {rttm_file}")
    speaker_ts = []
    try:
        with open(rttm_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(' ')
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split('_')[-1])])
        logger.info(f"RTTM файл успешно прочитан, обработано {len(speaker_ts)} записей")
        return speaker_ts
    except IOError as e:
        logger.error(f"Ошибка при чтении RTTM файла: {e}")
        raise

def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option='start'):
    logger.info("Начало сопоставления слов и спикеров...")
    s, e, sp = spk_ts[0]
    prev_spk = sp

    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    
    for wrd_dict in wrd_ts:
        try:
            ws = int(wrd_dict['start'] * 1000)
            we = int(wrd_dict['end'] * 1000)
            wrd = wrd_dict['text']
        except (KeyError, ValueError) as error:
            logger.warning(f"Пропускаем некорректную запись: {wrd_dict}. Ошибка: {error}")
            continue

        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
        
        wrd_spk_mapping.append({'word': wrd, 'start_time': ws, 'end_time': we, 'speaker': sp})
    
    logger.info(f"Сопоставление завершено, обработано {len(wrd_spk_mapping)} слов")
    return wrd_spk_mapping

def get_word_ts_anchor(s, e, option='start'):
    if option == 'end':
        return e
    elif option == 'mid':
        return (s + e) / 2
    return s

def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    logger.info("Начало формирования предложений с привязкой к спикерам...")
    s, e, spk = spk_ts[0]
    prev_spk = spk

    sentences = []
    current_sentence = {'speaker': f'Speaker {spk}', 'start_time': s, 'end_time': e, 'text': ''}

    for word_dict in word_speaker_mapping:
        word, spk = word_dict['word'], word_dict['speaker']
        s, e = word_dict['start_time'], word_dict['end_time']
        if spk != prev_spk:
            sentences.append(current_sentence)
            current_sentence = {'speaker': f'Speaker {spk}', 'start_time': s, 'end_time': e, 'text': ''}
        else:
            current_sentence['end_time'] = e
        current_sentence['text'] += word + ' '
        prev_spk = spk

    sentences.append(current_sentence)
    logger.info(f"Сформировано {len(sentences)} предложений.")
    return sentences

def get_speaker_aware_transcript(sentences_speaker_mapping, output_file):
    logger.info(f"Сохранение транскрипции с учетом спикеров в файл {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence_dict in sentences_speaker_mapping:
                sp = sentence_dict['speaker']
                text = sentence_dict['text']
                start_time = format_time(sentence_dict['start_time'] / 1000)  
                end_time = format_time(sentence_dict['end_time'] / 1000)  
                f.write(f'\n\n[{start_time} - {end_time}] {sp}: {text}')
        logger.info("Транскрипция успешно сохранена.")
    except IOError as e:
        logger.error(f"Ошибка при сохранении транскрипции: {e}")
        raise

def format_time(seconds):
    """Преобразует секунды в формат ЧЧ:ММ:СС."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

@app.post("/transcribe")
async def transcribe_audio_file(file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            base_file_name = os.path.splitext(file.filename)[0]
            wav_file_path = convert_to_wav(file_path, base_file_name)
            
            transcription_file = os.path.join(temp_dir, f'{base_file_name}_transcription.json')
            results = transcribe_audio(wav_file_path, transcription_file)
            
            manifest_path = os.path.join(temp_dir, f'{base_file_name}_manifest.json')
            prepare_diarization_manifest(wav_file_path, manifest_path)
            
            model_config = download_model_config()
            config = configure_diarizer(model_config, manifest_path)
            perform_diarization(config)
            
            speaker_ts = read_rttm_file(wav_file_path)
            word_ts = [{"text": word["word"], "start": word["start"], "end": word["end"]} 
                       for segment in results["segments"] 
                       for word in segment["words"]]
            
            wsm = get_words_speaker_mapping(word_ts, speaker_ts)
            wsm = apply_punctuation(wsm)
            ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
            
            output_file = os.path.join(temp_dir, f'{base_file_name}_diarized_transcript.txt')
            get_speaker_aware_transcript(ssm, output_file)
            
            with open(output_file, 'r', encoding='utf-8') as f:
                diarized_text = f.read()
            
            return JSONResponse(content={"diarized_text": diarized_text})
    
    except Exception as e:
        logger.error(f"Ошибка при обработке аудио: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def prepare_diarization_manifest(audio_file, output_file):
    logging.info(f"Подготовка манифеста диаризации для {audio_file}")
    base_file_name = os.path.splitext(os.path.basename(audio_file))[0]
    diarize_manifest = {
        'audio_filepath': audio_file,
        'offset': 0,
        'duration': None,
        'label': "infer",
        'text': "-",
        'num_speakers': None,
        'rttm_filepath': os.path.join(os.path.dirname(audio_file), f'{base_file_name}.rttm'),
        'uniq_id': ""
    }
    try:
        with open(output_file, 'w') as f:
            json.dump(diarize_manifest, f)
        logging.info(f"Манифест диаризации сохранен в {output_file}")
    except IOError as e:
        logging.error(f"Ошибка при сохранении манифеста диаризации: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv('PORT', 8345))
    uvicorn.run(app, host="0.0.0.0", port=port)