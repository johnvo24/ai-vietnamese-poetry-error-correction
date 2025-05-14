import sys
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from oauth2client.service_account import ServiceAccountCredentials
import httplib2
import os
import io
import torch
from tqdm import tqdm

class GDrive():
  def __init__(self):
    # Drive API scope
    self.SCOPES = ['https://www.googleapis.com/auth/drive']
    self.SERVICE_ACCOUNT_KEY_FILE   = 'service_account_key.json'
    self.service = self.get_drive_service()
    self.root = 'api_home'

  def get_credential(self):
    """Creates a Credential object with OAuth2 authorization."""
    credential = ServiceAccountCredentials.from_json_keyfile_name(
      self.SERVICE_ACCOUNT_KEY_FILE, self.SCOPES)
    if not credential or credential.invalid:
      print('Unable to authenticate using service account key.')
      sys.exit()
    return credential

  def get_drive_service(self):
    """Creates a service endpoint for Google Drive API."""
    http_auth = self.get_credential().authorize(httplib2.Http())
    return build('drive', 'v3', http=http_auth)
  
  def get_folder_id(self, folder_name, parent_folder_id=None):
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    if parent_folder_id: query += f" and '{parent_folder_id}' in parents"
    results = self.service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    if files: return files[0]['id']
    return None
  
  def create_folder(self, folder_name, parent_folder_id=None):
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_folder_id: file_metadata["parents"] = [parent_folder_id]
    folder = self.service.files().create(body=file_metadata, fields="id").execute()
    print(f"[SUC-gdrive] {folder_name} folder has been created successfully")
    return folder.get("id")
  
  def ensure_folder_exists(self, folder_path):
    path_parts = folder_path.split("/")
    parent_folder_id = self.get_folder_id(self.root)  # Bắt đầu từ root
    for folder_name in path_parts:
      folder_id = self.get_folder_id(folder_name, parent_folder_id)
      if not folder_id: folder_id = self.create_folder(folder_name, parent_folder_id)
      parent_folder_id = folder_id
    print(f"[INF-gdrive] {folder_path} path is ready!")
    return parent_folder_id

  def check_folder_exists(self, folder_path):
    """ex: temp, temp/test, ..."""
    path_parts = folder_path.split("/")
    parent_folder_id = self.get_folder_id(self.root)  # Bắt đầu từ root
    for folder_name in path_parts:
      folder_id = self.get_folder_id(folder_name, parent_folder_id)
      if not folder_id: 
        print(f"[WAR-gdrive] {folder_path} path doesn't exist")
        return None
      parent_folder_id = folder_id
    print(f"[INF-gdrive] {folder_path} path is ready")
    return parent_folder_id
  
  def list_all(self):
    file_list = []
    page_token = None
    while True:
      response = self.service.files().list(
        q = "trashed=false", # Only list non-deleted files
        fields = "nextPageToken, files(id, name)",
        pageToken = page_token
      ).execute()
      for file in response.get('files', []):
         file_list.append({'id': file['id'], 'name': file['name']})
      page_token = response.get('nextPageTonken', None)
      if page_token is None:
         break
    return file_list
  
  def get_file_id(self, file_name, folder_path):
    folder_id = self.check_folder_exists(folder_path[1:])
    if not folder_id: return None
    query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
    response = self.service.files().list(q=query, fields="files(id, name)").execute()
    files = response.get('files', [])
    if not files:
      print(f"[WAR] {file_name} doesn't exist in {folder_path} path")
      return None
    return files[0]['id']

  def upload_file(self, file_name, file_path, folder_path=f"/temp"):
    """Upload a file to Google Drive."""
    try:
      file_metadata = {'name': file_name}
      folder_id = self.ensure_folder_exists(folder_path[1:])
      file_metadata['parents'] = [folder_id]

      media = MediaFileUpload(file_path)
      file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
      print(f"[SUC-gdrive] File '{file_name}' uploaded with ID: {file.get('id')}")
      return file.get('id')
    except Exception as e:
      return
    
  def delete_files(self, file_ids):
    for file_id in file_ids:
      try:
        self.service.files().delete(fileId=file_id).execute()
        print(f"[SUC-gdrive] File deleted successfully")
      except Exception as e:
        print(f"[ERR-gdrive] File deletion failed: {e}")

  def download_file_to_memory(self, file_id):
    """Download a file from Google Drive and load it into memory."""
    request = self.service.files().get_media(fileId=file_id)
    file_data = io.BytesIO()
    downloader = MediaIoBaseDownload(file_data, request)
    # get size of file
    file_metadata = self.service.files().get(fileId=file_id, fields='size').execute()
    file_size = int(file_metadata.get('size', 0))
    # downloading
    done = False
    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
      while not done:
        status, done = downloader.next_chunk()
        if status:
          pbar.update(status.resumable_progress - pbar.n)
    file_data.seek(0)
    return file_data
  
  def download_file(self, file_id, destination_path):
    """Download a file from Google Drive"""
    file_data = self.download_file_to_memory(file_id)
    with open(destination_path, 'wb') as f:
      f.write(file_data.getvalue())
    print(f"[SUC-gdrive] File was downloaded and saved at: {destination_path}")

  def load_model_from_drive(self, file_name, model_dir):
    file_id = self.get_file_id(file_name=file_name, folder_path=f"/models/{model_dir}")
    file_data = self.download_file_to_memory(file_id)
    checkpoint = torch.load(file_data)
    return checkpoint
  
# lists = [{'id': '1oz60Akt7zCJKGkUE-VVcy2sYo-2xQEf-', 'name': 'best_checkpoint.tar'}, {'id': '1pLt_vnH32Xx-omae8QqLuoPmKGOAEW3K', 'name': 'best_checkpoint.tar'}, {'id': '1G63pU9rrueSjthjkaz-LKGOceovnBbsL', 'name': 'best_checkpoint.tar'}, {'id': '1pvl8uFFul-8R49327ltbqUaFSXPdfcVF', 'name': 'best_checkpoint.tar'}, {'id': '1RFVrd3smHjE0r9UIbjhz5HuTwu7Vus9M', 'name': 'best_checkpoint.tar'}, {'id': '10YR-Q5vwwru5cnzPnoZpiioLPnG0Ltor', 'name': 'phobert_gpt2'}, {'id': '1JQnMIdw52b8R63c8EsoWH1DrHFI2SEm1', 'name': 'models'}, {'id': '1E47XSQXwtDZClfNyrQk9dgrzvUkVuiLv', 'name': 'timer.py'}, {'id': '185a5lXlb_P6xd6Ivo5WWGGq689ae2rPW', 'name': 'timer.py'}, {'id': '1EhSMeEejc4sRkVr4ihCphBLyFXq8B1oQ', 'name': 'temp'}, {'id': '1IE5LIU24y53IOCEnVU2AH2dOtvepyljq', 'name': 'timer.py'}]
# gdrive = GDrive()
# gdrive.delete_files([list['id'] for list in lists])
# print(gdrive.list_all())
# gdrive.upload_file(file_name='timer.py', file_path=os.path.join(os.getcwd(), "models/checkpoints/sp_gpt2/best_checkpoint.tar"))
# print(gdrive.get_file_id('best_checkpoint.tar', '/models/phobert_gpt2'))