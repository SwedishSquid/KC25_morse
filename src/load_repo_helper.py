# run in kaggle to fetch repo

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

GITHUB_TOKEN = user_secrets.get_secret("GITHUB_MORSE_TOKEN")
USER = "SwedishSquid"
REPO_NAME = 'KC25_morse'
CLONE_URL = f"https://{USER}:{GITHUB_TOKEN}@github.com/{USER}/{REPO_NAME}.git"
get_ipython().system(f"git clone {CLONE_URL}")

import sys
sys.path.append("/kaggle/working/KC25_morse/src")

import morse
