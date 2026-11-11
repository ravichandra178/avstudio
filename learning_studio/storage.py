import io
from pathlib import Path

from .config import Settings


class DocumentStorage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.local_path = Path(settings.local_storage_path)
        if settings.s3_bucket:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required when S3_BUCKET is configured; "
                    "unset S3_BUCKET for local filesystem storage"
                ) from exc
            self.client = boto3.client("s3", region_name=settings.aws_region)

    def put(self, document_id: str, filename: str, content: bytes) -> str:
        key = f"documents/{document_id}/{Path(filename).name}"
        if self.client is not None:
            self.client.upload_fileobj(io.BytesIO(content), self.settings.s3_bucket, key)
        else:
            target = self.local_path / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return key

    def get(self, key: str) -> bytes:
        if self.client is not None:
            response = self.client.get_object(Bucket=self.settings.s3_bucket, Key=key)
            return response["Body"].read()
        target = self.local_path / key
        if not target.is_file():
            raise FileNotFoundError(f"Stored document not found: {key}")
        return target.read_bytes()
