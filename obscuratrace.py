#!/usr/bin/env python3
#!/usr/bin/env python3
# privacyvision_ultimate.py

# --- Automatic Dependency Check and Installer (Runs before any other imports) ---
import importlib.util
import subprocess
import sys, os

# Required dependencies and their import module names
dependencies = {
    "insightface": "insightface",
    "onnxruntime": "onnxruntime",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "requests": "requests",
    "beautifulsoup4": "bs4",
    "yt-dlp": "yt_dlp",
    "PyYAML": "yaml",
    "colorama": "colorama",
    "schedule": "schedule",
    "tqdm": "tqdm"
}

# Check which dependencies are missing
missing_packages = []
for pkg, module_name in dependencies.items():
    if importlib.util.find_spec(module_name) is None:  # if module not found
        missing_packages.append(pkg)

# If any are missing, prompt user to install
if missing_packages:
    print(f"Missing dependencies detected: {', '.join(missing_packages)}.")
    choice = input("Install automatically? (Y/N): ").strip().lower()
    if choice not in ("y", "yes"):
        # User chose not to install
        sys.exit("Error: Required dependencies are missing. Exiting.")
    # Proceed to install missing packages
    print(f"Installing missing packages: {', '.join(missing_packages)}...")
    try:
        # Use current Python interpreter to install packages&#8203;:contentReference[oaicite:4]{index=4}
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
    except Exception as e:
        sys.exit(f"Installation failed: {e}")
    # After successful installation, restart the script to load new libraries&#8203;:contentReference[oaicite:5]{index=5}
    print("Dependencies installed. Restarting script...", flush=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)
# --- End of dependency check ---

# Now import the dependencies (they should all be installed at this point)
import insightface
import onnxruntime
import cv2               # from opencv-python
import numpy as np       # numpy
import requests
from bs4 import BeautifulSoup  # beautifulsoup4
import yaml              # PyYAML
from colorama import Fore, Style
import schedule
from tqdm import tqdm

# ... (rest of the original script logic goes here) ...
# The main functionality of privacyvision_ultimate.py continues below.
# For example, function and class definitions, main execution code, etc.
#!/usr/bin/env python3
"""
PrivacyVision Ultimate - Single File
Enhancements:
 1) Colorful, dynamic text-based UI with ASCII flair
 2) Hourly auto-check for new videos
 3) Parallel extraction / speed optimization
 4) Extended search to Brazzers, YouJizz, Pornhub, etc.
 5) Better result visualization
 6) Lightweight UI approach
"""

import os
import sys
import time
import yaml
import requests
import urllib.parse
import multiprocessing
import cv2
import colorama
import schedule
import yt_dlp
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from colorama import Fore, Style
from insightface.app import FaceAnalysis

# Initialize colorama for Windows compatibility
colorama.init()

CONFIG_FILE = "config.yaml"

DEFAULT_CONFIG = {
    "face_recognition": {"threshold": 0.6},
    "instagram_users": [],
    "profile_images_dir": "profiles",
    "videos_dir": "videos",
    "frames_dir": "frames",
    "report_dir": "reports",
    "search_sites": ["pornhub", "youjizz", "brazzers"],  # placeholder
    "videos": []
}

###########################
#   LOAD & SAVE CONFIG
###########################
def load_config():
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = yaml.safe_load(f)
        for k, v in DEFAULT_CONFIG.items():
            if k not in data:
                data[k] = v
        return data
    else:
        return DEFAULT_CONFIG.copy()

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(cfg, f)

###########################
#   SEARCH MODULES
###########################
def search_pornhub(query, max_pages=1):
    base_url = "https://www.pornhub.com"
    encoded_q = urllib.parse.quote_plus(query)
    results = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for page in range(1, max_pages+1):
        url = f"{base_url}/video/search?search={encoded_q}&page={page}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Example naive selector
                links = soup.select("a.videoPreviewBg") or soup.select(".pcVideoListItem a[href^='/view_video.php']")
                for link in links:
                    href = link.get("href", "")
                    if href.startswith("/view_video.php"):
                        results.append(base_url + href)
        except Exception as e:
            print(f"{Fore.RED}Error searching Pornhub: {e}{Style.RESET_ALL}")
    return list(set(results))

def search_youjizz(query, max_pages=1):
    # Example placeholder; real YouJizz scraping logic may differ
    base_url = "https://www.youjizz.com"
    encoded_q = urllib.parse.quote_plus(query)
    results = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for page in range(1, max_pages+1):
        url = f"{base_url}/search/?q={encoded_q}&page={page}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Selector for links
                links = soup.select("a.list-videos__snippet")
                for link in links:
                    href = link.get("href", "")
                    if href.startswith("/videos/"):
                        results.append(base_url + href)
        except Exception as e:
            print(f"{Fore.RED}Error searching YouJizz: {e}{Style.RESET_ALL}")
    return list(set(results))

def search_brazzers(query, max_pages=1):
    # Placeholder; brazzers likely requires a more advanced approach or subscription check
    base_url = "https://www.brazzers.com"
    results = []
    headers = {"User-Agent": "Mozilla/5.0"}
    # Brazzers might not have a public search page in the same sense. This is a naive placeholder.
    for page in range(1, max_pages+1):
        # e.g. hypothetical URL: /search?q=<query>
        url = f"{base_url}/search?q={urllib.parse.quote_plus(query)}&page={page}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # placeholder link detection
                links = soup.select("a[href^='/scenes/']")
                for link in links:
                    href = link.get("href", "")
                    if href.startswith("/scenes/"):
                        results.append(base_url + href)
        except Exception as e:
            print(f"{Fore.RED}Error searching Brazzers: {e}{Style.RESET_ALL}")
    return list(set(results))

def unified_search(query, sites, max_pages=1):
    """
    Perform a search for the given query across multiple adult sites:
    'pornhub', 'youjizz', 'brazzers', etc.
    Return a combined list of all found video URLs.
    """
    all_results = []
    for site in sites:
        print(f"{Fore.CYAN}Searching {site} for '{query}'...{Style.RESET_ALL}")
        if site == "pornhub":
            found = search_pornhub(query, max_pages)
        elif site == "youjizz":
            found = search_youjizz(query, max_pages)
        elif site == "brazzers":
            found = search_brazzers(query, max_pages)
        else:
            print(f"Site '{site}' not implemented. Skipping.")
            found = []
        print(f" Found {len(found)} links in {site}.")
        all_results.extend(found)
    return list(set(all_results))

###########################
#  VIDEO DOWNLOAD & EXTRACT
###########################
def download_video(url, output_dir="videos", prefix="video"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'outtmpl': f'{output_dir}/{prefix}.%(ext)s',
        'format': 'best[ext=mp4]/best'
    }
    path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
    except Exception as e:
        print(f"{Fore.RED}Download error: {e}{Style.RESET_ALL}")
    return path

def extract_frames(params):
    """ Worker function for multiprocessing. """
    video_path, frames_output_dir, fps = params
    if not video_path or not os.path.isfile(video_path):
        return 0
    os.makedirs(frames_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_count = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = fps
    frame_interval = max(1, int(round(video_fps / fps)))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_name = os.path.join(frames_output_dir, f"{base_name}_frame{frame_idx}.jpg")
            cv2.imwrite(out_name, frame)
            saved_count += 1
        frame_idx += 1
    cap.release()
    return saved_count

def process_videos(video_list, frames_dir="frames", videos_dir="videos", fps=1):
    """ Download each URL if needed, then extract frames in parallel. """
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    # Step 1: Download videos
    downloaded_paths = []
    for idx, url in enumerate(video_list, start=1):
        if url.startswith("http"):
            v_path = download_video(url, videos_dir, prefix=f"video_{idx}")
            if v_path: 
                downloaded_paths.append(v_path)
        elif os.path.isfile(url):
            downloaded_paths.append(url)
        else:
            print(f"{Fore.YELLOW}Skipping invalid path/URL: {url}{Style.RESET_ALL}")

    # Step 2: Multiprocessing extract frames
    tasks = [(v_path, frames_dir, fps) for v_path in downloaded_paths]
    results = []
    with multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
        results = list(tqdm(pool.imap(extract_frames, tasks), total=len(tasks), desc="Extracting frames"))
    # Print summary
    for path, count in zip(downloaded_paths, results):
        print(f"Extracted {count} frames from {path}")

###########################
#  INSTAGRAM SCRAPER
###########################
def download_profile_images(usernames, output_dir="profiles"):
    os.makedirs(output_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    for username in usernames:
        profile_url = f"https://www.instagram.com/{username}/"
        try:
            r = requests.get(profile_url, headers=headers, timeout=10)
            if r.status_code != 200:
                print(f"{Fore.RED}Failed to fetch {username} (HTTP {r.status_code}){Style.RESET_ALL}")
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            og_img = soup.find("meta", property="og:image")
            if not og_img:
                print(f"{Fore.YELLOW}No og:image for {username}{Style.RESET_ALL}")
                continue
            pic_url = og_img.get("content", "")
            if not pic_url:
                continue
            data = requests.get(pic_url, headers=headers, timeout=10).content
            ext = ".jpg"
            if ".png" in pic_url.lower(): 
                ext = ".png"
            save_path = os.path.join(output_dir, username + ext)
            with open(save_path, "wb") as f:
                f.write(data)
            print(f"{Fore.GREEN}Saved {username} -> {save_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error scraping {username}: {e}{Style.RESET_ALL}")

###########################
#   ARC FACE RECOGNITION
###########################
class FaceRecognizer:
    def __init__(self, model_name='buffalo_l', threshold=0.6):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640,640))
        self.threshold = threshold
        self.known_faces = []  # (name, embedding)

    def add_face(self, person_name, image_path):
        img = cv2.imread(image_path)
        if img is None: return
        faces = self.app.get(img)
        if not faces: return
        # pick largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.normed_embedding
        self.known_faces.append((person_name, emb))

    def identify_in_image(self, img):
        results = []
        faces = self.app.get(img)
        for face in faces:
            emb = face.normed_embedding
            bbox = face.bbox.astype(int)
            for (kname, known_emb) in self.known_faces:
                sim = float(np.dot(emb, known_emb))
                if sim >= self.threshold:
                    results.append((kname, sim, bbox))
        return results

###########################
#    RESULT VISUALIZATION
###########################
def generate_report(matches, report_dir="reports"):
    """
    Save match results to a CSV-like text file for easy reading.
    """
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"match_report_{int(time.time())}.txt")
    with open(report_file, "w") as f:
        f.write("FRAME,NAME,SIMILARITY,BBOX(x1,y1,x2,y2)\n")
        for frame_path, name, sim, bbox in matches:
            x1,y1,x2,y2 = bbox
            f.write(f"{frame_path},{name},{sim:.2f},({x1},{y1},{x2},{y2})\n")
    print(f"{Fore.MAGENTA}Report saved: {report_file}{Style.RESET_ALL}")

###########################
#   HOURLY AUTOMATION
###########################
def hourly_check(config):
    # Example: re-run the search & face recognition every hour
    query = input(f"{Fore.CYAN}Enter name/surname/metatag to search automatically:{Style.RESET_ALL} ")
    sites = config.get("search_sites", ["pornhub"])
    found = unified_search(query, sites, max_pages=1)
    if found:
        print(f"{Fore.GREEN}Discovered {len(found)} new links{Style.RESET_ALL}")
        config["videos"].extend(found)
        config["videos"] = list(set(config["videos"]))
        save_config(config)
    else:
        print(f"{Fore.YELLOW}No new videos found for {query}.{Style.RESET_ALL}")
    # Then download & process
    process_videos(config["videos"], config["frames_dir"], config["videos_dir"], fps=1)
    # Then run face recognition if you want
    run_face_recognition(config)

###########################
#  MAIN UI
###########################
def run_face_recognition(config):
    # Load known faces from config["profile_images_dir"]
    th = config["face_recognition"].get("threshold", 0.6)
    recognizer = FaceRecognizer(threshold=th)
    pdir = config["profile_images_dir"]
    if not os.path.isdir(pdir):
        print(f"{Fore.YELLOW}No profile images dir: {pdir}{Style.RESET_ALL}")
        return
    # Add known faces
    for file in os.listdir(pdir):
        if file.lower().endswith((".png",".jpg",".jpeg")):
            recognizer.add_face(os.path.splitext(file)[0], os.path.join(pdir, file))
    if not recognizer.known_faces:
        print(f"{Fore.YELLOW}No known faces loaded, skipping detection.{Style.RESET_ALL}")
        return

    frames_dir = config["frames_dir"]
    if not os.path.isdir(frames_dir):
        print(f"{Fore.YELLOW}No frames dir found: {frames_dir}{Style.RESET_ALL}")
        return

    all_matches = []
    for root, dirs, files in os.walk(frames_dir):
        for file in tqdm(files, desc="Scanning frames"):
            if file.lower().endswith((".png",".jpg",".jpeg")):
                frame_path = os.path.join(root, file)
                img = cv2.imread(frame_path)
                if img is None: 
                    continue
                detections = recognizer.identify_in_image(img)
                for (name, sim, bbox) in detections:
                    print(f"{Fore.GREEN}[MATCH] {name} in {file} (sim={sim:.2f}){Style.RESET_ALL}")
                    all_matches.append((frame_path, name, sim, bbox))

    if all_matches:
        # Save a nice report
        generate_report(all_matches, config.get("report_dir","reports"))
    else:
        print(f"{Fore.BLUE}No matches found in any frame.{Style.RESET_ALL}")

def fancy_banner():
    print(f"{Fore.YELLOW}{Style.BRIGHT}")
    print("     ____  ____  _______  __________  ___  ____  __")
    print("    / __ \\/ __ \\/ ____/ |/ / ____/ / / / |/ / / / /")
    print("   / /_/ / / / / __/  |   / __/ / / / /    / / / / ")
    print("  / _, _/ /_/ / /___ /   / /___/ /_/ / /|  / /_/ /  ")
    print(" /_/ |_|\\____/_____//_/|_/_____/\\____/_/ |_/\\____/   (Ultimate)")
    print("======================================================")
    print("   => Powered by ArcFace, multi-site scraping, & GPU <=")
    print(f"{Style.RESET_ALL}")


def menu():
    fancy_banner()
    print(f"{Fore.MAGENTA}{Style.BRIGHT}")
    print("========= PrivacyVision Ultimate Menu =========")
    print(f"{Style.RESET_ALL}")
    print(f"{Fore.CYAN}[1]{Style.RESET_ALL} Search adult sites for query (Brazzers/YouJizz/Pornhub)")
    print(f"{Fore.CYAN}[2]{Style.RESET_ALL} Download & process videos (multi-process extraction)")
    print(f"{Fore.CYAN}[3]{Style.RESET_ALL} Scrape Instagram profiles (update known faces)")
    print(f"{Fore.CYAN}[4]{Style.RESET_ALL} Run face recognition on extracted frames")
    print(f"{Fore.CYAN}[5]{Style.RESET_ALL} Start hourly auto-check (continually) [BETA]")
    print(f"{Fore.CYAN}[6]{Style.RESET_ALL} Quit")
    print("============================================")
    choice = input(f"{Fore.YELLOW}Select an option: {Style.RESET_ALL}")
    return choice.strip()

def main_loop():
    if not os.path.isfile(CONFIG_FILE):
        cfg = DEFAULT_CONFIG.copy()
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(cfg, f)
        print(f"{Fore.GREEN}Created default config.yaml{Style.RESET_ALL}")

    while True:
        config = load_config()
        choice = menu()
        if choice == "1":
            query = input(f"{Fore.GREEN}Enter your query: {Style.RESET_ALL}")
            sites = config.get("search_sites", ["pornhub"])
            pages_str = input("Max pages to scrape [1]? ")
            try:
                max_pages = int(pages_str)
            except:
                max_pages = 1
            found = unified_search(query, sites, max_pages)
            print(f"Found {len(found)} total new links.")
            if found:
                save_input = input("Save these links to config? [y/N]: ")
                if save_input.lower().startswith("y"):
                    config["videos"].extend(found)
                    config["videos"] = list(set(config["videos"]))
                    save_config(config)
        elif choice == "2":
            if not config.get("videos"):
                print(f"{Fore.YELLOW}No videos in config.yaml => videos{Style.RESET_ALL}")
            else:
                fps_inp = input("Frames per second [1]? ")
                try:
                    fps = float(fps_inp)
                except:
                    fps = 1
                process_videos(config["videos"], config["frames_dir"], config["videos_dir"], fps)
        elif choice == "3":
            if config["instagram_users"]:
                download_profile_images(config["instagram_users"], config["profile_images_dir"])
            else:
                print("No instagram_users in config. Please add them in config.yaml.")
        elif choice == "4":
            run_face_recognition(config)
        elif choice == "5":
            print(f"{Fore.GREEN}Starting hourly schedule... Press Ctrl+C to stop.{Style.RESET_ALL}")
            schedule.every().hour.do(hourly_check, config=config)
            # We block here in a loop
            while True:
                schedule.run_pending()
                time.sleep(1)
        elif choice == "6":
            print(f"{Fore.CYAN}Exiting. Bye!{Style.RESET_ALL}")
            sys.exit(0)
        else:
            print(f"{Fore.RED}Invalid choice, try again.{Style.RESET_ALL}")


if __name__ == "__main__":
    main_loop()
