import tempfile
import base64
from pathlib import Path
from flask import current_app
from werkzeug.utils import secure_filename
import os
from services.image_loader import load_image_any
from services.file_validation import validate_upload_file


class CustomUploadResult:
    def __init__(self, is_valid: bool, error: str = None, file_obj=None, save_path=None, filename=None, user_name="Patient", language="English", is_volume=False):
        self.is_valid = is_valid
        self.error = error
        self.file_obj = file_obj
        self.save_path = save_path
        self.filename = filename
        self.user_name = user_name
        self.language = language
        self.is_volume = is_volume

def is_allowed_file(filename: str) -> tuple[bool, str]:
    if not isinstance(filename, str) or '.' not in filename:
        return False, "Invalid file format or missing extension."
    ext = filename.rsplit('.', 1)[1].lower()
    allowed_exts = current_app.config.get('ALLOWED_EXTENSIONS', {'jpg', 'jpeg', 'png', 'dcm', 'dicom'})
    if ext not in allowed_exts:
        return False, f"Unsupported file extension: .{ext}"
    return True, ""

def parse_any_upload(request, save_dir: Path, enforce_filename: str = None) -> CustomUploadResult:
    save_path = ""
    filename = enforce_filename or ""
    user_name = "Patient"
    language = "English"

    # 1. Raw Binary
    if 'image/' in request.content_type:
        ext = request.content_type.split('/')[-1] if '/' in request.content_type else 'jpg'
        fd, save_path = tempfile.mkstemp(suffix=f".{ext}", dir=save_dir)
        with os.fdopen(fd, 'wb') as f:
            f.write(request.data)
        filename = enforce_filename or getattr(request, 'filename', f"upload.{ext}")
        user_name = request.args.get("user_name", "Patient")
        language = request.args.get("language", "English")
        
        validation = validate_upload_file(Path(save_path), filename)
        if not validation.ok:
            try:
                os.remove(save_path)
            except Exception:
                pass
            return CustomUploadResult(False, validation.reason)

        _, is_volume = load_image_any(Path(save_path), return_meta=True)
        return CustomUploadResult(True, save_path=save_path, filename=filename, user_name=user_name, language=language, is_volume=is_volume)

    # 2. JSON Base64
    if request.is_json:
        data = request.get_json()
        if 'file_base64' in data:
            try:
                b64_str = data['file_base64']
                if not b64_str: return CustomUploadResult(False, "Empty base64")
                file_data = base64.b64decode(b64_str)
                filename = filename or data.get('filename', 'upload.jpg')
                ext = filename.split('.')[-1] if '.' in filename else 'jpg'
                fd, save_path = tempfile.mkstemp(suffix=f".{ext}", dir=save_dir)
                with os.fdopen(fd, 'wb') as f:
                    f.write(file_data)
                user_name = data.get("user_name", "Patient")
                language = data.get("language", "English")
                
                validation = validate_upload_file(Path(save_path), filename)
                if not validation.ok:
                    try:
                        os.remove(save_path)
                    except Exception:
                        pass
                    return CustomUploadResult(False, validation.reason)

                _, is_volume = load_image_any(Path(save_path), return_meta=True)
                return CustomUploadResult(True, save_path=save_path, filename=filename, user_name=user_name, language=language, is_volume=is_volume)
            except Exception as e:
                return CustomUploadResult(False, str(e))

    # 3. Multipart Form-Data
    file_obj = None
    if 'file' in request.files: file_obj = request.files['file']
    elif 'image' in request.files: file_obj = request.files['image']
    
    if file_obj:
        if file_obj.filename == '': return CustomUploadResult(False, "No selected file")
        filename = filename or secure_filename(file_obj.filename)
        is_valid, reason = is_allowed_file(filename)
        if not is_valid: return CustomUploadResult(False, reason)
        
        save_path = str(save_dir / filename)
        file_obj.save(save_path)
        user_name = request.form.get("user_name", "Patient")
        language = request.form.get("language", "English")
        
        validation = validate_upload_file(Path(save_path), filename)
        if not validation.ok:
            try:
                os.remove(save_path)
            except Exception:
                pass
            return CustomUploadResult(False, validation.reason)

        _, is_volume = load_image_any(Path(save_path), return_meta=True)
        return CustomUploadResult(True, save_path=save_path, filename=filename, user_name=user_name, language=language, is_volume=is_volume)

    return CustomUploadResult(False, "No valid image data provided")
