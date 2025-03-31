# VTT Translator

A Python script that translates VTT (and SRT) subtitle files using the Gemini API. This tool automates the process of translating subtitles, making video content more accessible to a wider audience.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Error Handling and Logging](#error-handling-and-logging)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Automatic Translation:** Translates subtitle files from an unknown source language to a specified target language using the Gemini API.
- **SRT to VTT Conversion:** Automatically converts SRT subtitle files to VTT format before translation.
- **Batch Processing:** Processes subtitles in batches to optimize API usage and performance.
- **Concurrency:** Utilizes multi-threading to speed up the translation process.
- **Retry Mechanism:** Implements a retry mechanism to handle API errors and ensure translation completion.
- **Configuration Validation:** Uses Pydantic for configuration validation, ensuring that settings are correct before execution.
- **Backup System:** Creates a backup of the original subtitle file before overwriting it with the translated version.
- **Detailed Logging:** Provides detailed logging to track the progress of the translation and troubleshoot issues.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd vtt-translator
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    If you don't have a `requirements.txt` file, create one with the following content:

    ```
    python-dotenv
    webvtt-converter
    pydantic
    google-generative-ai
    ```

## Configuration

The script uses environment variables and a `Config` class (using Pydantic) to manage configuration.

1.  **Set up your Gemini API key:**

    - Obtain an API key from the [Google AI Studio](https://aistudio.google.com/apikey).
    - Set the `GEMINI_API_KEY` environment variable. The recommended way is to create a `.env` file in the project root directory with the following content:

      ```
      GEMINI_API_KEY=YOUR_API_KEY
      ```

2.  **Customize other settings (optional):**

    You can customize the following settings by modifying the default values in the `Config` class:

    - `MODEL`: The Gemini model to use for translation (default: `gemini-2.0-flash-lite`). // Fast with excelent RPM
    - `TARGET_LANGUAGE`: The target language for translation (default: `"Portuguese (Brazilian)"`). // My language
    - `TRANSLATION_CONTEXT`: Contextual information for the translation (default: `"This is a video subtitle."`). // Help Gemini to understand your subtitle content
    - `TEMPERATURE`: The temperature setting for the Gemini API (default: `0.5`). // Good for translations
    - `BATCH_SIZE`: The number of subtitle entries to process in each batch (default: `200`). // Secure value without exceeding the output tokens.
    - `MAX_RETRIES`: The maximum number of retry attempts for failed translations (default: `3`).
    - `RETRY_DELAY`: The delay (in seconds) between retry attempts (default: `2`).
    - `MAX_WORKERS`: The maximum number of worker threads for concurrent processing (default: `5`).

## Usage

1.  **Run the script:**

    ```bash
    python vtt_translator.py [directory]
    ```

    - `directory` (optional): The directory containing the VTT/SRT files to translate. If not specified, the script will process files in the current working directory.

2.  **The script will:**

    - Find all SRT and VTT files in the specified directory (or the current directory if none is specified).
    - Convert any SRT files to VTT format.
    - Create a backup of each original VTT file.
    - Translate the subtitles in each VTT file using the Gemini API.
    - Save the translated subtitles to the original VTT file, overwriting the original content.

## Code Structure

The script is organized into several classes and functions:

- **Constants:** Defines constants such as file extensions, backup suffixes, and default values.
- **`Config` (Pydantic Model):** Manages and validates configuration settings.
- **Logging Configuration:** Sets up logging for the script.
- **Prompt Template:** Defines the prompt template used to interact with the Gemini API.
- **`SubtitleEntry` (Pydantic Model):** Represents a single subtitle entry with `id`, `start`, `end`, and `caption` fields.
- **`TranslatedItem` (Pydantic Model):** Represents a translated subtitle item with `id` and `caption` fields.
- **Utility Functions:**
  - `create_backup(file_path: str) -> str`: Creates a backup of the original file.
- **`VttProcessor` Class:**
  - `find_srt_and_vtt_files(directory: str) -> Tuple[List[str], List[str]]`: Finds SRT and VTT files in a directory.
  - `convert_srt_to_vtt(srt_file_path: str) -> str`: Converts an SRT file to VTT format.
  - `parse_subtitle_data(vtt_file_path: str) -> List[SubtitleEntry]`: Parses subtitle data from a VTT file.
- **`TranslationService` Class:**
  - `__init__(self, config: Config)`: Initializes the translation service with the given configuration.
  - `count_tokens(self, text: str) -> int`: Counts tokens in a text using the Gemini API.
  - `_translate_batch(self, batch: List[SubtitleEntry], batch_num: int, total_batches: int, is_retry_batch: bool, filename: str) -> Tuple[List[TranslatedItem], Set[int]]`: Translates a batch of subtitle entries.
  - `translate(self, subtitle_data: List[SubtitleEntry], filename: str) -> List[TranslatedItem]`: Translates a list of subtitle entries.
- **`VttWriter` Class:**
  - `save_translated_vtt(original_file_path: str, translated_data: List[TranslatedItem]) -> str`: Saves translated data to a VTT file, backing up the original.
- **`main()` Function:** Orchestrates the translation process.

## Error Handling and Logging

The script includes comprehensive error handling and logging to help diagnose and resolve issues.

- **Configuration Validation:** The `Config` class uses Pydantic validators to ensure that configuration settings are valid.
- **Error Handling:** The script uses `try...except` blocks to catch and handle exceptions that may occur during file processing, API calls, and data validation.
- **Logging:** The script uses the `logging` module to record detailed information about the translation process, including:
  - Informational messages about file processing and translation progress.
  - Warning messages about potential issues, such as missing files or incomplete translations.
  - Error messages about failures, such as API errors or validation errors.
  - Debug messages with detailed traceback information for troubleshooting.

The log level can be configured using the `LOG_LEVEL` environment variable.

## Contributing

Contributions are welcome! Please feel free to submit pull requests with bug fixes, new features, or improvements to the documentation.

## License

This project is licensed under the [MIT License](LICENSE).
