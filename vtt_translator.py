import os
import sys
import webvtt
import json
import logging
import time
import traceback
import concurrent.futures
from typing import List, Dict, Any, Optional, Set, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, TypeAdapter, field_validator
from google import genai
from google.genai import types

# Constants - Define all constants at the top for easy modification
SRT_EXTENSION = '.srt'
VTT_EXTENSION = '.vtt'
BACKUP_SUFFIX = '_original'
VTT_HEADER = "WEBVTT\n\n"
API_RESPONSE_MIME_TYPE = 'application/json'
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_MODEL = "gemini-2.0-flash-lite"
DEFAULT_TARGET_LANGUAGE = "Portuguese (Brazilian)"
DEFAULT_TRANSLATION_CONTEXT = "This is a video subtitle."
DEFAULT_TEMPERATURE = 0.5
DEFAULT_BATCH_SIZE = 200
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2
DEFAULT_MAX_WORKERS = 5

load_dotenv()

# Configuration using Pydantic BaseModel for validation and type hinting
class Config(BaseModel):
    API_KEY: str = os.getenv("GEMINI_API_KEY") # Load from environment variable
    MODEL: str = DEFAULT_MODEL
    TARGET_LANGUAGE: str = DEFAULT_TARGET_LANGUAGE
    TRANSLATION_CONTEXT: str = DEFAULT_TRANSLATION_CONTEXT
    TEMPERATURE: float = DEFAULT_TEMPERATURE
    BATCH_SIZE: int = DEFAULT_BATCH_SIZE
    MAX_RETRIES: int = DEFAULT_MAX_RETRIES
    RETRY_DELAY: int = DEFAULT_RETRY_DELAY
    MAX_WORKERS: int = DEFAULT_MAX_WORKERS

    @field_validator("API_KEY")
    def api_key_must_be_set(cls, v):
        if not v:
            raise ValueError("GEMINI_API_KEY environment variable must be set.")
        return v

    @field_validator("TEMPERATURE")
    def temperature_must_be_between_0_and_1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1.")
        return v

    @field_validator("BATCH_SIZE")
    def batch_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Batch size must be positive.")
        return v

    @field_validator("MAX_RETRIES")
    def max_retries_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Max retries must be positive.")
        return v

    @field_validator("RETRY_DELAY")
    def retry_delay_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Retry delay must be positive.")
        return v

    @field_validator("MAX_WORKERS")
    def max_workers_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Max workers must be positive.")
        return v

# Initialize Configuration
try:
    config = Config()
except ValidationError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)

# Logging Configuration
LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), DEFAULT_LOG_LEVEL)
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vtt_translator")

# Suppress noisy loggers
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Prompt Template - Using a separate file or a more robust templating engine is better for complex templates
prompt_template = """
## Task: Translate Captions from Unknown Source Language

**Objective:**
Translate the 'caption' field for each object in the provided JSON input array from the specified **Source Language** to the **Target Language**.

**Source Language is unique but unknown, identify the input language through 5 caption samples.**

**What are you translating?**
Use this context to help you: {translation_context}
File name: {filename}

**Target Language:**
{target_language}

**Input Data:**
An array of JSON objects, where all 'caption' fields are in the **Source Language**. Each object has the structure:
{{ "id": number, "caption": string }}

The specific input array to process is:
{input_array_json_string}

**Instructions:**
1. Iterate through each object in the input array.
2. Translate the text in the "caption" field directly from the identified **Source Language** into the **Target Language** ({target_language}).
3. **Crucially, maintain the original line break structure (`\\n`) in the translation.** If a caption has multiple lines separated by `\\n`, the translation MUST preserve those line breaks in the corresponding semantic locations.
4. Construct a new JSON array as output.
5. Each object in the output array MUST contain the *original* "id" (as a number) and the *translated* "caption" (as a string).
6. The output MUST be ONLY the JSON array, strictly conforming to the required schema: `list[{{\n  "id": number,\n  "caption": string\n}}]`.

**Process the entire input array provided above and return the resulting JSON array.**

"""

# Initialize Gemini Client - Only if API Key is available
client = None
try:
    client = genai.Client(api_key=config.API_KEY)
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    sys.exit(1)

# Data Models - Using Pydantic for data validation
class SubtitleEntry(BaseModel):
    id: int
    start: str
    end: str
    caption: str

class TranslatedItem(BaseModel):
    id: int
    caption: str
    start: Optional[str] = None  # Add start and end for easier handling later
    end: Optional[str] = None

# Utility Functions
def create_backup(file_path: str) -> str:
    """Creates a backup of the original file."""
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    file_base, file_ext = os.path.splitext(file_name)
    backup_file_path = os.path.join(file_dir, f"{file_base}{BACKUP_SUFFIX}{file_ext}")

    try:
        os.rename(file_path, backup_file_path)
        logger.info(f"Original file renamed to: {backup_file_path}")
        return backup_file_path
    except OSError as e:
        logger.error(f"Error creating backup of {file_path}: {e}")
        return None

# Classes
class VttProcessor:

    @staticmethod
    def find_srt_and_vtt_files(directory: str) -> Tuple[List[str], List[str]]:
        """Finds SRT and VTT files in a directory."""
        srt_files = []
        vtt_files = []

        try:
            for root_dir, subdirs, files in os.walk(directory):
                for file_name in files:
                    file_path = os.path.join(root_dir, file_name)
                    if file_name.lower().endswith(SRT_EXTENSION):
                        srt_files.append(file_path)
                    elif file_name.lower().endswith(VTT_EXTENSION):
                        vtt_files.append(file_path)

            logger.info(f"Found {len(srt_files)} SRT files and {len(vtt_files)} VTT files")
            return srt_files, vtt_files

        except Exception as e:
            logger.error(f"Error searching for files: {e}")
            return [], []

    @staticmethod
    def convert_srt_to_vtt(srt_file_path: str) -> str:
        """Converts an SRT file to VTT format."""
        try:
            vtt_file_path = os.path.splitext(srt_file_path)[0] + VTT_EXTENSION
            vtt = webvtt.from_srt(srt_file_path)
            vtt.save(vtt_file_path)
            logger.info(f"SRT file converted successfully: {srt_file_path} -> {vtt_file_path}")
            return vtt_file_path
        except Exception as e:
            logger.error(f"Error converting SRT file to VTT: {e}")
            raise

    @staticmethod
    def parse_subtitle_data(vtt_file_path: str) -> List[SubtitleEntry]:
        """Parses subtitle data from a VTT file."""
        subtitle_entries = []
        try:
            for entry_index, caption_entry in enumerate(webvtt.read(vtt_file_path), 1):
                try:
                    subtitle_entry = SubtitleEntry(
                        id=entry_index,
                        start=caption_entry.start,
                        end=caption_entry.end,
                        caption=caption_entry.text
                    )
                    subtitle_entries.append(subtitle_entry)
                except ValidationError as e:
                    logger.error(f"Validation error for entry {entry_index}: {e}")

            logger.info(f"Extracted {len(subtitle_entries)} subtitle entries from file")
            return subtitle_entries
        except Exception as e:
            logger.error(f"Error parsing subtitle file: {e}")
            return []

class TranslationService:

    def __init__(self, config: Config):
        self.config = config

    def count_tokens(self, text: str) -> int:
        """Counts tokens in a text using the Gemini API."""
        try:
            response = client.models.count_tokens(
                model=self.config.MODEL,
                contents=text
            )
            return response.total_tokens
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0

    def _translate_batch(self, batch: List[SubtitleEntry], batch_num: int, total_batches: int, is_retry_batch: bool, filename: str) -> Tuple[List[TranslatedItem], Set[int]]:
        """Translates a batch of subtitle entries."""
        retry_count = 0
        batch_ids = {item.id for item in batch}

        log_prefix = f"Batch {'retry ' if is_retry_batch else ''}{batch_num}"
        if not is_retry_batch:
            log_prefix += f" of {total_batches}"

        while retry_count < self.config.MAX_RETRIES:
            try:
                if retry_count > 0:
                    logger.info(f"Attempt {retry_count + 1} of {self.config.MAX_RETRIES} for {log_prefix}")
                    time.sleep(self.config.RETRY_DELAY * retry_count)

                # Convert SubtitleEntry objects to dictionaries for JSON serialization
                batch_data = [item.model_dump() for item in batch]
                gemini_input = json.dumps(batch_data, indent=2, ensure_ascii=False)

                final_prompt = prompt_template.format(
                    target_language=self.config.TARGET_LANGUAGE,
                    input_array_json_string=gemini_input,
                    translation_context=self.config.TRANSLATION_CONTEXT,
                    filename=filename
                )

                logger.info(f"Sending prompt to Gemini for {log_prefix}.")

                response = client.models.generate_content(
                    model=self.config.MODEL,
                    contents=final_prompt,
                    config={
                        'response_mime_type': API_RESPONSE_MIME_TYPE,
                        'response_schema': list[TranslatedItem],
                        'temperature': self.config.TEMPERATURE,
                    }
                )

                output_tokens = self.count_tokens(response.text)
                logger.info(f"Used {output_tokens} output tokens for {log_prefix}.")

                if not response.text:
                    raise ValueError("Empty response received from API")

                parsed_response = json.loads(response.text)
                translated_item_adapter = TypeAdapter(List[TranslatedItem])
                validated_data: List[TranslatedItem] = translated_item_adapter.validate_python(parsed_response)
                batch_translated_data = [item.model_dump() for item in validated_data]

                translated_ids = {item['id'] for item in batch_translated_data}
                missing_ids = batch_ids - translated_ids

                if not missing_ids:
                    logger.info(f"{log_prefix} translated successfully: {len(batch_translated_data)} items")
                    return batch_translated_data, set()
                else:
                    logger.warning(f"In {log_prefix}, incomplete response. Missing IDs: {missing_ids} (attempt {retry_count + 1})")
                    retry_count += 1
                    if retry_count >= self.config.MAX_RETRIES:
                        logger.error(f"Failed to obtain all IDs for {log_prefix} after {self.config.MAX_RETRIES} attempts.")
                        return [], batch_ids
                    continue

            except (json.JSONDecodeError, ValidationError, ValueError, types.generation_types.BlockedPromptException, Exception) as e:
                retry_count += 1
                if isinstance(e, ValidationError):
                    logger.error(f"Pydantic validation error in {log_prefix} (attempt {retry_count}): {e}")
                elif isinstance(e, types.generation_types.BlockedPromptException):
                    logger.error(f"Prompt blocked in {log_prefix} (attempt {retry_count}): {e}")
                else:
                    logger.error(f"Error translating {log_prefix} (attempt {retry_count}): {e}")
                logger.debug(traceback.format_exc())

                if retry_count >= self.config.MAX_RETRIES:
                    logger.error(f"Failure after {self.config.MAX_RETRIES} attempts for {log_prefix} due to error.")
                    return [], batch_ids
                continue

        logger.error(f"Retry loop for {log_prefix} completed unexpectedly.")
        return [], batch_ids

    def translate(self, subtitle_data: List[SubtitleEntry], filename: str) -> List[TranslatedItem]:
        """Translates a list of subtitle entries."""
        all_translated_items = []
        untranslated_ids_overall = {item.id for item in subtitle_data}
        processed_ids = set()

        retry_batch_loop_count = 0

        subtitle_data_by_id = {item.id: item for item in subtitle_data}

        while untranslated_ids_overall and retry_batch_loop_count <= self.config.MAX_RETRIES:
            current_ids_to_process = list(untranslated_ids_overall)
            data_to_process = [subtitle_data_by_id[id] for id in current_ids_to_process]
            total_entries_in_loop = len(data_to_process)
            batch_size = self.config.BATCH_SIZE

            is_retry_loop = retry_batch_loop_count > 0
            if is_retry_loop:
                logger.info(f"--- Starting retry cycle #{retry_batch_loop_count} for {total_entries_in_loop} IDs ---")

            newly_untranslated_ids_in_loop = set()

            batches = []
            for i in range(0, total_entries_in_loop, batch_size):
                batch_data = data_to_process[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches_in_loop = (total_entries_in_loop + batch_size - 1) // batch_size

                if not is_retry_loop:
                    logger.info(f"Preparing batch {batch_num} of {total_batches_in_loop} "
                                f"(subtitles {i + 1}-{min(i + batch_size, total_entries_in_loop)} of {len(subtitle_data)})")

                batches.append((batch_data, batch_num, total_batches_in_loop, is_retry_loop, filename))

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                future_to_batch = {
                    executor.submit(self._translate_batch, *batch_params): batch_params
                    for batch_params in batches
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_params = future_to_batch[future]
                    batch_data, batch_num, total_batches, is_retry, filename = batch_params

                    try:
                        translated_batch_items, untranslated_in_batch = future.result()

                        all_translated_items.extend(translated_batch_items)

                        processed_ids.update(item['id'] for item in translated_batch_items)

                        logger.info(f"Batch {batch_num}: {len(translated_batch_items)} items translated, {len(untranslated_in_batch)} failures")

                    except Exception as e:
                        logger.error(f"Uncaught error in batch processing {batch_num}: {e}")
                        logger.debug(traceback.format_exc())

                        batch_ids = {item.id for item in batch_data}
                        newly_untranslated_ids_in_loop.update(batch_ids)

            untranslated_ids_overall = untranslated_ids_overall.intersection(newly_untranslated_ids_in_loop)

            if untranslated_ids_overall:
                retry_batch_loop_count += 1
                logger.info(f"Remaining {len(untranslated_ids_overall)} untranslated IDs for the next cycle")
                if retry_batch_loop_count > self.config.MAX_RETRIES:
                    logger.warning(f"Maximum retry cycles ({self.config.MAX_RETRIES}) reached.")
            else:
                logger.info("All items translated successfully.")

        if untranslated_ids_overall:
            logger.warning(f"Could not translate {len(untranslated_ids_overall)} IDs after all attempts: {untranslated_ids_overall}")

        logger.info(f"Translation complete: {len(all_translated_items)} items translated in total.")

        # Incorporate timestamps from original SubtitleEntry objects
        final_translated_items = []
        for item in all_translated_items:
            original_item = subtitle_data_by_id.get(item['id'])
            if original_item:
                # Remover start/end se eles já existirem no item para evitar duplicação
                item_data = {k: v for k, v in item.items() if k not in ['start', 'end']}
                
                # Adicionar start/end do original
                item_data['start'] = original_item.start
                item_data['end'] = original_item.end
                
                # Criar o objeto TranslatedItem
                translated_item = TranslatedItem(**item_data)
                final_translated_items.append(translated_item)
            else:
                logger.warning(f"Translated ID {item['id']} not found in original data to add timestamp.")

        logger.info("Timestamps added to translated data.")
        return final_translated_items

class VttWriter:

    @staticmethod
    def save_translated_vtt(original_file_path: str, translated_data: List[TranslatedItem]) -> str:
        """Saves translated data to a VTT file, backing up the original."""
        try:
            # Sort translated data by ID
            sorted_data = sorted(translated_data, key=lambda x: x.id)

            # Create a backup of the original file
            backup_file_path = create_backup(original_file_path)
            if not backup_file_path:
                return ""

            translated_file_path = original_file_path

            with open(translated_file_path, 'w', encoding='utf-8') as f:
                f.write(VTT_HEADER)

                for entry in sorted_data:
                    f.write(f"{entry.start} --> {entry.end}\n")
                    f.write(f"{entry.caption}\n\n")

            logger.info(f"Translated VTT file saved to original location: {translated_file_path}")
            return translated_file_path

        except Exception as e:
            logger.error(f"Error saving translated VTT file: {e}")
            logger.debug(traceback.format_exc())
            return ""

def main():
    """Main function to orchestrate the translation process."""
    logger.info("Starting file processing")

    try:
        search_directory = os.getcwd()
        if len(sys.argv) > 1:
            search_directory = sys.argv[1]

        if not os.path.isdir(search_directory):
            logger.error(f"Error: The directory '{search_directory}' does not exist.")
            return 1

        vtt_processor = VttProcessor()
        srt_files, vtt_files = vtt_processor.find_srt_and_vtt_files(search_directory)

        # Convert SRT files to VTT
        for srt_file in srt_files:
            try:
                vtt_path = vtt_processor.convert_srt_to_vtt(srt_file)
                if vtt_path:
                    vtt_files.append(vtt_path)
            except Exception as e:
                logger.error(f"Error converting SRT {srt_file}: {e}")

        if not vtt_files:
            logger.warning(f"No VTT files found or converted in: {search_directory}")
            return 1

        logger.info("VTT files for processing:")
        for i, file_path in enumerate(vtt_files, 1):
            logger.info(f"{i}. {file_path}")

        # Process each VTT file
        translation_service = TranslationService(config)
        vtt_writer = VttWriter()

        for vtt_file_path in vtt_files:
            try:
                logger.info(f"\n=== Processing file: {vtt_file_path} ===")

                # Extract subtitle data
                subtitle_data = vtt_processor.parse_subtitle_data(vtt_file_path)
                if not subtitle_data:
                    logger.warning(f"No subtitle data extracted from: {vtt_file_path}")
                    continue

                logger.info(f"Starting translation to {config.TARGET_LANGUAGE}")

                filename = os.path.basename(vtt_file_path)  # Getting the filename
                translated_data = translation_service.translate(subtitle_data, filename)

                if not translated_data:
                    logger.error(f"Failed to translate subtitles for: {vtt_file_path}")
                    continue

                # Save the translated VTT file
                translated_file_path = vtt_writer.save_translated_vtt(vtt_file_path, translated_data)

                if translated_file_path:
                    logger.info(f"Translation complete: {len(translated_data)} subtitles translated")
                    logger.info(f"File saved to: {translated_file_path}")

            except Exception as e:
                logger.error(f"Error processing file {vtt_file_path}: {e}")
                logger.debug(traceback.format_exc())

        return 0

    except Exception as e:
        logger.error(f"Uncaught error: {e}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())