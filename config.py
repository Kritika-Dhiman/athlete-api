from dotenv import load_dotenv
import os


class Config(object):
    SECRET_KEY=os.getenv('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY is not set. Please configure it in .env")

load_dotenv()  # Load environment variables from .env

secret_key = os.getenv("SECRET_KEY")
database_url = os.getenv("DATABASE_URL")

print("Secret Key:", secret_key)  # Just to check if it's working
