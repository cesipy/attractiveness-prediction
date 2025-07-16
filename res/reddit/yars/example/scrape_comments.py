import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from yars.yars import YARS
from yars.utils import display_results


miner = YARS()

post_url = "https://www.reddit.com/r/Innsbruck/comments/1m09lll/motorrad_geklaut/"

# Scrape post details using its permalink
permalink = post_url.split('reddit.com')[1]
post_details = miner.scrape_post_details(permalink)
if post_details:
    display_results(post_details, "POST DATA")
else:
    print("Failed to scrape post details.")

