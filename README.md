ObscuraTrace

ObscuraTrace is a privacy protection tool that scans adult content platforms to detect unauthorized appearances of your face or identity. Acting as your personal digital investigator, it automates hourly searches across multiple adult sites to swiftly identify new uploads. Featuring advanced facial recognition, metadata analysis, and high-performance frame processing, ObscuraTrace notifies you whenever it detects a match—helping you address potential privacy concerns promptly.

Key Features
	1.	AI-Powered Facial Recognition
	•	Uses ArcFace (InsightFace) to recognize your face accurately, even under varying conditions and angles.
	2.	Multi-Platform Scanning
	•	Monitors Pornhub, YouJizz, Brazzers, and other major adult sites to ensure broad, real-time coverage.
	3.	Automated Hourly Monitoring
	•	Performs unattended scans every hour, minimizing the chance of missing newly uploaded content.
	4.	Metadata & Contextual Analysis
	•	Examines video titles, tags, and descriptions to detect your name, username, or other potentially identifying data.
	5.	Instagram Profile Integration
	•	Leverages Instagram profile images to improve face matching precision, catching subtle re-uploads.
	6.	High-Performance Frame Processing
	•	Rapidly extracts and analyzes frames from long videos, supporting near-real-time detection even on large datasets.
	7.	Real-Time Alerts & Reporting
	•	Immediately notifies you of matches, producing a human-readable report with video timestamps, source links, and confidence scores.

Installation

Prerequisites
	1.	Python 3.7+ (Linux or macOS recommended; supports Apple Silicon and Intel).
	2.	Git (optional) if cloning from a repository.

Automatic Dependency Installation

ObscuraTrace includes an auto-installer for missing libraries. On startup, it checks for required dependencies and, if absent, prompts you to install them. No manual pip installation is necessary unless you prefer it.

The core dependencies are:
	•	insightface
	•	onnxruntime
	•	opencv-python
	•	numpy
	•	requests
	•	beautifulsoup4
	•	yt-dlp
	•	PyYAML
	•	colorama
	•	schedule
	•	tqdm

Setup Steps
	1.	Clone or Download

git clone https://github.com/YourUser/ObscuraTrace.git
cd ObscuraTrace

Or download the ZIP file, extract it, and cd into the directory.

	2.	Run the Script

python privacyvision_ultimate.py

	•	If dependencies are missing, you’ll be prompted to confirm their installation.

	3.	Config File
	•	If config.yaml isn’t found, ObscuraTrace creates a default.
	•	Tweak fields like instagram_users, face_recognition.threshold, or search_sites if desired.

Usage

ObscuraTrace presents a menu on startup:
	1.	Search Adult Sites – Provide a name or keyword to find relevant videos.
	2.	Download & Process Videos – Downloads the videos (if they’re URLs) and extracts frames for analysis.
	3.	Scrape Instagram Profiles – Updates your known faces using the provided IG usernames.
	4.	Run Face Recognition – Compares extracted frames against known faces, reporting any matches.
	5.	Start Hourly Auto-Check – Continuously re-check for new content every hour.
	6.	Quit – Exit ObscuraTrace.

Hourly Scanning
	•	Select option 5 to start an hourly cycle automatically searching for content and scanning new videos.
	•	Press Ctrl + C at any time to interrupt continuous scanning.

Report Generation
	•	When face recognition identifies a match, a detailed text-based report is stored in report_dir (default reports/).
	•	Each match entry includes the frame path, the recognized face name, similarity score, and bounding box or timestamps.

Configuration

By default, the script reads from config.yaml:

instagram_users:
  - "example_user"
face_recognition:
  threshold: 0.65
search_sites:
  - "pornhub"
  - "youjizz"
  - "brazzers"
profile_images_dir: "profiles"
videos_dir: "videos"
frames_dir: "frames"
report_dir: "reports"
videos: []

	•	instagram_users: A list of IG accounts to pull images from.
	•	search_sites: Adult platforms to check (e.g., Pornhub, YouJizz, Brazzers).
	•	threshold: Cosine similarity cutoff for face matching, 0.0–1.0 (higher is stricter).
	•	videos: Pre-existing video URLs or local file paths, if you have them.

Modify these values to suit your needs.

Performance Tips
	1.	GPU Acceleration:
	•	ObscuraTrace attempts GPU usage (ctx_id=0) if available. For Apple Silicon or dedicated GPUs, ensure the environment is set up properly for maximum speed.
	2.	Frame Extraction Rate:
	•	Adjust the frames-per-second (FPS) extraction in the code or at runtime for fewer frames (faster) vs. thorough scanning.
	3.	Parallel Processing:
	•	Frame extraction uses multiprocessing by default, speeding up large-scale analysis.

Privacy & Ethical Usage

Disclaimer: ObscuraTrace is designed to protect users’ privacy by detecting unauthorized appearances. Scraping adult sites or analyzing personal data can carry legal and ethical considerations:
	•	Use only with explicit consent from the individual whose face is monitored.
	•	Comply with site Terms of Service and privacy regulations (e.g., GDPR).
	•	Disable or avoid features if they might violate another person’s privacy or relevant laws.

Troubleshooting
	1.	Dependency Installation Failure
	•	If auto-install fails, run:

pip install insightface onnxruntime opencv-python numpy requests beautifulsoup4 yt-dlp PyYAML colorama schedule tqdm


	2.	Encountering CAPTCHAs
	•	Some adult sites enforce CAPTCHAs for high-volume scraping. Slow down search intervals or use a more advanced scraping setup if needed.
	3.	Low Face Detection
	•	Ensure your reference images (from Instagram) are clear. If ArcFace can’t detect a face in the reference, recognition won’t work.

Contributing
	•	Issues: Submit bug reports or feature requests via GitHub Issues.
	•	Pull Requests: Fork the repo, commit improvements, and open a PR.
	•	Testing: Provide logs or test data for new features to accelerate merges.

License

Distributed under the MIT License. See LICENSE for details.

Keep your identity safe—let ObscuraTrace reveal the hidden dangers, so you can act swiftly.
