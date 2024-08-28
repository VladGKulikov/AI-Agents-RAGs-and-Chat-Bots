import tqdm
from  youtube_transcript_api import YouTubeTranscriptApi
from pytube import Playlist

# YouTube playlist URL
# Stanford CS224N: Natural Language Processing with Deep Learning | 2023
playlist_url = 'https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4'


summary_prompt = '''I give you transcription youtube video - summarize it in 1500-2000 symbols, 
mark main questions and points.

TRANSCRIPTION:
{transcription}
'''

def get_yt_transcription(from_yt_video_id, to_txt_file):
    transcript = YouTubeTranscriptApi.get_transcript(from_yt_video_id)

    with open(to_txt_file, 'w') as file:
        for line in tqdm.tqdm(transcript):
            file.write(line['text']+'\n')

# Save the video IDs from playlist to a file
def get_yt_video_ids(from_playlist_url, to_txt_file):
    # Create a Playlist object
    playlist = Playlist(from_playlist_url)

    # Get all video IDs
    video_ids = [video.video_id for video in playlist.videos]

    with open(to_txt_file, 'w') as file:
        for video_id in video_ids:
            file.write(video_id + '\n')

if __name__ == '__main__':

    video_ids_file = 'playlist_video_ids.txt'        

    # Create a Playlist object
    playlist = Playlist(playlist_url)

    # Get all video IDs
    video_ids = [video.video_id for video in playlist.videos]    
    
    for i, video_id in enumerate(video_ids):
        print(f'Load transcription from youtube video #{i+1}:')
        get_yt_transcription(video_id, f'data/youtube/yt_{i:02}_' + f'{video_id}' +'.txt')


 