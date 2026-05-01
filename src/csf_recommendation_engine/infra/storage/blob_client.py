from azure.storage.blob import BlobServiceClient


class BlobStore:
    def __init__(self, connection_string: str, container: str) -> None:
        if not connection_string:
            raise ValueError("Azure storage connection string is required")
        self._service = BlobServiceClient.from_connection_string(connection_string)
        self._container = self._service.get_container_client(container)

    async def download_blob_bytes(self, blob_path: str) -> bytes:
        downloader = self._container.download_blob(blob_path)
        return downloader.readall()

    async def upload_blob_bytes(self, blob_path: str, payload: bytes, overwrite: bool = True) -> None:
        self._container.upload_blob(name=blob_path, data=payload, overwrite=overwrite)
