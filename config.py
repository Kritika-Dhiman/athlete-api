from dotenv import load_dotenv
import os


load_dotenv()

class Config(object):
    SECRET_KEY = os.getenv("SECRET_KEY")
    MONGO_URI = os.getenv("MONGO_URL")  

    if not SECRET_KEY:
        raise ValueError("SECRET_KEY is not set. Please configure it in .env")


print("Secret Key:", Config.SECRET_KEY)
print("Database URL:", Config.MONGO_URI)
