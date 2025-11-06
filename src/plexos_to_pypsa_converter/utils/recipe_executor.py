"""Execute recipe instructions to download and organize model files."""

import platform
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import rarfile
import requests
from tqdm import tqdm

from plexos_to_pypsa_converter.db.registry import MODEL_REGISTRY

# Constants
SEPARATOR_LINE = "=" * 70
DOWNLOAD_CHUNK_SIZE = 8192
REQUEST_TIMEOUT = 30


class RecipeExecutionError(Exception):
    """Raised when recipe execution fails."""


class RecipeExecutor:
    """Executes recipe instructions to download and organize model files."""

    def __init__(self, model_id: str, model_dir: Path, verbose: bool = True):
        self.model_id = model_id
        self.model_dir = Path(model_dir)
        self.temp_dir = self.model_dir / ".download_temp"
        self.verbose = verbose
        self.steps_completed: list[int] = []
        self._step_handlers = {
            "download": self._step_download,
            "extract": self._step_extract,
            "move": self._step_move,
            "copy": self._step_copy,
            "rename": self._step_rename,
            "delete": self._step_delete,
            "create_dir": self._step_create_dir,
            "flatten": self._step_flatten,
            "validate": self._step_validate,
            "manual": self._step_manual,
        }

    def execute_recipe(self, recipe: list[dict[str, Any]]) -> bool:
        """Execute all steps in recipe."""
        if not recipe:
            if self.verbose:
                print("Empty recipe provided - nothing to execute")
            return True

        i = 0
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)

            total_steps = len(recipe)
            if self.verbose:
                self._print_header(total_steps)

            for i, instruction in enumerate(recipe, 1):
                step_type = instruction["step"]
                description = instruction.get("description", f"Step {i}: {step_type}")

                if self.verbose:
                    print(f"[{i}/{total_steps}] {description}")

                self._execute_step(instruction)
                self.steps_completed.append(i)

            if self.verbose:
                self._print_success()

        except Exception as e:
            if self.verbose:
                self._print_error(i, len(recipe), e)

            msg = f"Recipe execution failed at step {i}/{len(recipe)}: {e}"
            raise RecipeExecutionError(msg) from e

        else:
            return True

        finally:
            self._cleanup_temp()

    def _print_header(self, total_steps: int) -> None:
        """Print recipe execution header."""
        print(f"\n{SEPARATOR_LINE}")
        print(f"Executing recipe for model: {self.model_id}")
        print(f"Total steps: {total_steps}")
        print(f"{SEPARATOR_LINE}\n")

    def _print_success(self) -> None:
        """Print success message."""
        print(f"\n{SEPARATOR_LINE}")
        print(" Recipe completed successfully!")
        print(f"Model installed to: {self.model_dir}")
        print(f"{SEPARATOR_LINE}\n")

    def _print_error(self, step: int, total_steps: int, error: Exception) -> None:
        """Print error message."""
        print(f"\n{SEPARATOR_LINE}")
        print(f"Recipe failed at step {step}/{total_steps}")
        print(f"Error: {error}")
        print(f"{SEPARATOR_LINE}\n")

    def _execute_step(self, instruction: dict[str, Any]) -> None:
        """Execute a single instruction."""
        step_type = instruction["step"]
        handler = self._step_handlers.get(step_type)

        if handler is None:
            available_types = ", ".join(self._step_handlers.keys())
            msg = f"Unknown step type: {step_type}. Available types: {available_types}"
            raise ValueError(msg)

        handler(instruction)

    def _step_download(self, instruction: dict[str, Any]) -> None:
        """Handle download step."""
        url = instruction["url"]
        target_filename = instruction["target"]
        method = instruction.get("method", "direct")

        target_path = self.model_dir / target_filename

        if method == "direct":
            self._download_direct(url, target_path)
        elif method == "post":
            form_data = instruction.get("form_data", {})
            self._download_post(url, target_path, form_data)
        else:
            msg = f"Unsupported download method: {method}"
            raise ValueError(msg)

    def _step_extract(self, instruction: dict[str, Any]) -> None:
        """Handle extract step."""
        source = self.model_dir / instruction["source"]
        target = self.model_dir / instruction["target"]
        archive_format = instruction.get("format")

        self._extract_archive(source, target, archive_format)

    def _step_move(self, instruction: dict[str, Any]) -> None:
        """Handle move step."""
        source_pattern = instruction["source"]
        target = self.model_dir / instruction["target"]
        sources = list(self.model_dir.glob(source_pattern))

        if not sources:
            if self.verbose:
                print(f"  Warning: No files matched pattern: {source_pattern}")
            return

        target.mkdir(parents=True, exist_ok=True)
        for source_path in sources:
            shutil.move(str(source_path), str(target / source_path.name))

    def _step_copy(self, instruction: dict[str, Any]) -> None:
        """Handle copy step."""
        source = self.model_dir / instruction["source"]
        target = self.model_dir / instruction["target"]

        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)

    def _step_rename(self, instruction: dict[str, Any]) -> None:
        """Handle rename step."""
        source = self.model_dir / instruction["source"]
        target = self.model_dir / instruction["target"]

        if source.exists():
            source.rename(target)

    def _step_delete(self, instruction: dict[str, Any]) -> None:
        """Handle delete step."""
        pattern = instruction["pattern"]
        recursive = instruction.get("recursive", False)
        paths = list(
            self.model_dir.rglob(pattern) if recursive else self.model_dir.glob(pattern)
        )

        for path in paths:
            if path.is_file():
                path.unlink()
            elif path.is_dir() and recursive:
                shutil.rmtree(path)

    def _step_create_dir(self, instruction: dict[str, Any]) -> None:
        """Handle create_dir step."""
        path = self.model_dir / instruction["path"]
        path.mkdir(parents=instruction.get("parents", True), exist_ok=True)

    def _step_flatten(self, instruction: dict[str, Any]) -> None:
        """Handle flatten step."""
        source = self.model_dir / instruction["source"]
        levels = instruction.get("levels", 1)

        for _ in range(levels):
            if not source.is_dir():
                break

            for item in source.iterdir():
                shutil.move(str(item), str(source.parent / item.name))

            if source.exists() and not list(source.iterdir()):
                source.rmdir()

    def _step_validate(self, instruction: dict[str, Any]) -> None:
        """Handle validate step."""
        checks = instruction["checks"]
        model_config = MODEL_REGISTRY[self.model_id]

        for check in checks:
            if check == "xml_exists":
                xml_filename = model_config.get("xml_filename")
                if xml_filename:
                    xml_path = self.model_dir / xml_filename
                    if not xml_path.exists():
                        msg = f"Validation failed: XML file not found: {xml_filename}"
                        raise RecipeExecutionError(msg)

            elif check.startswith("min_files:"):
                min_count = int(check.split(":")[1])
                file_count = sum(1 for _ in self.model_dir.rglob("*") if _.is_file())
                if file_count < min_count:
                    msg = f"Validation failed: Expected at least {min_count} files, found {file_count}"
                    raise RecipeExecutionError(msg)

            elif check.startswith("required_dir:"):
                dir_name = check.split(":", 1)[1]
                dir_path = self.model_dir / dir_name
                if not dir_path.is_dir():
                    msg = f"Validation failed: Required directory not found: {dir_name}"
                    raise RecipeExecutionError(msg)

            elif check.startswith("file_exists:"):
                filename = check.split(":", 1)[1]
                file_path = self.model_dir / filename
                if not file_path.exists():
                    msg = f"Validation failed: Required file not found: {filename}"
                    raise RecipeExecutionError(msg)

            elif check.startswith("min_size_mb:"):
                min_size_mb = float(check.split(":")[1])
                total_size = sum(
                    f.stat().st_size for f in self.model_dir.rglob("*") if f.is_file()
                )
                total_size_mb = total_size / (1024 * 1024)
                if total_size_mb < min_size_mb:
                    msg = f"Validation failed: Expected at least {min_size_mb} MB, found {total_size_mb:.1f} MB"
                    raise RecipeExecutionError(msg)

        if self.verbose:
            print("   Validation passed")

    def _step_manual(self, instruction: dict[str, Any]) -> None:
        """Handle manual step - display instructions and wait for user."""
        instructions_text = instruction["instructions"]

        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 70)
        print(instructions_text)
        print("=" * 70)

        response = input(
            "\nPress Enter when files are in place (or 'cancel' to abort): "
        )

        if response.lower() == "cancel":
            msg = "Manual download cancelled by user"
            raise RecipeExecutionError(msg)

    def _download_direct(self, url: str, target: Path) -> None:
        """Download file from direct HTTP/HTTPS URL with progress bar."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.aemo.com.au/",
            }
            response = requests.get(
                url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with target.open("wb") as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"  Downloading {target.name}",
                    ) as pbar:
                        for chunk in response.iter_content(
                            chunk_size=DOWNLOAD_CHUNK_SIZE
                        ):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    # No content-length header
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)

        except requests.RequestException as e:
            msg = f"Download failed from {url}: {e}"
            raise RecipeExecutionError(msg) from e

    def _download_post(self, url: str, target: Path, form_data: dict[str, str]) -> None:
        """Download file via POST request with form data."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }

            response = requests.post(
                url,
                data=form_data,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                headers=headers,
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with target.open("wb") as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"  Downloading {target.name}",
                    ) as pbar:
                        for chunk in response.iter_content(
                            chunk_size=DOWNLOAD_CHUNK_SIZE
                        ):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    # No content-length header
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)

        except requests.RequestException as e:
            msg = f"POST download failed from {url}: {e}"
            raise RecipeExecutionError(msg) from e

    def _extract_archive(
        self, archive_path: Path, target_dir: Path, format: str | None = None
    ) -> None:
        """Extract archive file to target directory."""
        target_dir.mkdir(parents=True, exist_ok=True)

        if format is None:
            format = self._detect_archive_format(archive_path)

        try:
            if format == "zip":
                self._extract_zip(archive_path, target_dir)
            elif format in ("tar", "tar.gz", "tar.bz2"):
                self._extract_tar(archive_path, target_dir, format)
            elif format == "rar":
                self._extract_rar(archive_path, target_dir)
            else:
                msg = f"Unsupported archive format: {format}"
                raise RecipeExecutionError(msg)

        except (zipfile.BadZipFile, tarfile.TarError, rarfile.Error) as e:
            msg = f"Failed to extract archive {archive_path}: {e}"
            raise RecipeExecutionError(msg) from e

    def _detect_archive_format(self, archive_path: Path) -> str:
        """Detect archive format from file extension."""
        if archive_path.suffix == ".zip":
            return "zip"
        elif archive_path.name.endswith(".tar.gz") or archive_path.suffix == ".tgz":
            return "tar.gz"
        elif archive_path.name.endswith(".tar.bz2"):
            return "tar.bz2"
        elif archive_path.suffix == ".tar":
            return "tar"
        elif archive_path.suffix == ".rar":
            return "rar"
        else:
            msg = f"Cannot auto-detect archive format for: {archive_path}"
            raise RecipeExecutionError(msg)

    def _safe_extract_path(self, target_dir: Path, member_path: str) -> Path:
        """Validate extraction path to prevent path traversal attacks."""
        target_abs = target_dir.resolve()
        member_abs = (target_abs / member_path).resolve()

        try:
            member_abs.relative_to(target_abs)
        except ValueError as e:
            msg = (
                f"Archive member '{member_path}' would extract outside target "
                f"directory (potential path traversal attack)"
            )
            raise RecipeExecutionError(msg) from e

        return member_abs

    def _extract_zip(self, archive_path: Path, target_dir: Path) -> None:
        """Extract ZIP archive safely, preventing path traversal."""
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                extract_path = self._safe_extract_path(target_dir, member)

                if member.endswith("/"):
                    extract_path.mkdir(parents=True, exist_ok=True)
                else:
                    extract_path.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        zip_ref.open(member) as source,
                        extract_path.open("wb") as target,
                    ):
                        shutil.copyfileobj(source, target)

    def _extract_tar(self, archive_path: Path, target_dir: Path, format: str) -> None:
        """Extract TAR archive safely, preventing path traversal."""
        mode_map = {
            "tar": "r",
            "tar.gz": "r:gz",
            "tar.bz2": "r:bz2",
        }
        mode = mode_map[format]

        with tarfile.open(str(archive_path), mode) as tar_ref:  # type: ignore[call-overload]
            for member in tar_ref.getmembers():
                extract_path = self._safe_extract_path(target_dir, member.name)

                if member.isdir():
                    extract_path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    extract_path.parent.mkdir(parents=True, exist_ok=True)
                    with tar_ref.extractfile(member) as source:
                        if source:
                            with extract_path.open("wb") as target:
                                shutil.copyfileobj(source, target)

    def _extract_rar(self, archive_path: Path, target_dir: Path) -> None:
        """Extract RAR archive safely, preventing path traversal."""
        tool_found = shutil.which("unrar") or shutil.which("unar")

        if not tool_found:
            os_name = platform.system()

            install_instructions = {
                "Darwin": "Install with Homebrew:\n  brew install unrar",
                "Linux": "Install with package manager:\n  # Ubuntu/Debian:\n  sudo apt-get install unrar\n  # or use unar:\n  sudo apt-get install unar\n  \n  # Fedora/RHEL:\n  sudo yum install unrar",
                "Windows": "Download UnRAR from:\n  https://www.rarlab.com/rar_add.htm\n  and add it to your PATH",
            }

            instructions = install_instructions.get(
                os_name, "Install unrar or unar for your operating system"
            )

            msg = (
                f"RAR extraction requires 'unrar' or 'unar' command-line tool.\n\n"
                f"{instructions}\n\n"
                f"After installation, retry the model download."
            )
            raise RecipeExecutionError(msg)

        try:
            with rarfile.RarFile(str(archive_path), "r") as rar_ref:
                for member in rar_ref.infolist():
                    extract_path = self._safe_extract_path(target_dir, member.filename)

                    if member.isdir():
                        extract_path.mkdir(parents=True, exist_ok=True)
                    else:
                        extract_path.parent.mkdir(parents=True, exist_ok=True)
                        with (
                            rar_ref.open(member.filename) as source,
                            extract_path.open("wb") as target,
                        ):
                            shutil.copyfileobj(source, target)
        except rarfile.Error as e:
            msg = f"Failed to extract RAR archive: {e}"
            raise RecipeExecutionError(msg) from e

    def _cleanup_temp(self) -> None:
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to clean up temp directory: {e}")
