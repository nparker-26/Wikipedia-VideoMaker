# System imports
import sys
import os
import subprocess
import re
import random

# File management imports
import glob
from natsort import natsorted
import shutil
import string

# Wikipedia and NLP imports
import wikipedia
import nltk.data

# Stable Whisper (Subtitles)
import stable_whisper
import pysrt

# Image Management
from cropimage import Cropper
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from PIL import Image
import open_clip

# PyTorch
import torch

# Article Summarization
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from newspaper import Article

# OpenVoice Local Files
from se_extractor import *
from api import BaseSpeakerTTS, ToneColorConverter

def WikipediaSummaryGet(subject):
    summary = wikipedia.summary(subject, auto_suggest=False)
    summary = SummaryChanges(summary)
    print(summary)

    #split the paragraph into sentences
    summary = re.sub(r"\[.*?\]|\\(.*?\\)", "", summary) #removes brackets from article
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(summary)

    #Count the number of sentences to produce certain number of images
    ImageCount = int(len(sentences)*1.25)
    ImageCount = round(ImageCount)
    if ImageCount < 10:
        ImageCount = 10
    return summary, sentences, ImageCount

def URLSummaryGet(subject, path, url):
    if not os.path.exists(path):
        os.makedirs(path)
    article = Article(url)
    article.download()
    article.parse()
    with open(path + '\\' + subject + '.txt', "a", encoding="utf-8") as file: #TODO change encoding based on language
        article = article.text
        file.write(article)
    #path + '\\"' + subject + '".txt'
    with open(path + '\\' + subject + '.txt', "r", encoding="utf-8") as f:
    # Read all of the lines from the file
        lines = f.readlines()
    
    #clean the result obtained from the article parser
    while True:
        try:
            lines.remove('\n')
            lines = [re.sub(r'\[[^]]*\]', '', item) for item in lines] #remove everything in brackets for wikipedia references
            lines = [element.replace('\n', '') for element in lines]
            lines = [item for item in lines if item.endswith(".") or item.endswith("?") or item.endswith("!")]
        except ValueError:
            break

    for line in lines:
        print(line)
    text = " ".join(lines)

    #facebook bart text generation
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', model_max_length=512)
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn', return_dict=True)

    inputs = tokenizer.encode("summarize: " + text,
    return_tensors='pt',
    max_length=512,
    truncation=True)

    summary_ids = model.generate(inputs, max_length=150, min_length=125, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0])
    print('-------------------------------------')
    summary = re.sub(r'<[^>]+>', '', summary)
    print(summary)

    #split the paragraph into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.treebank.pickle')
    sentences = tokenizer.tokenize(summary)

    #Count the number of sentences to produce certain number of images
    ImageCount = int(len(sentences)*1.25)
    ImageCount = round(ImageCount)
    if ImageCount < 10:
        ImageCount = 10
    return summary, sentences, ImageCount


def ImproveSentences(sentences):
    first = sentences[0]
    try:
        #remove the parenthesis and the stuff in side them
        first = re.sub("\(.*\)", "", first)
        first = re.sub(' +', ' ', first)
        print(first)
        return first
    except:
        print("no dates")

#crop my image to 500X500 and center it around important / useful content
def CropImage(subject, path):
    cropper = Cropper()
    entries = glob(r'C:\\Users\\natha\Desktop\\Audviya\\' + subject +r'\\*.jpg')
    entries = natsorted(entries)

    # Set target_size to be a tuple (size, size), only square output is supported now
    for entry in entries:
        try:
            indexstr = str(entries.index(entry))
            result = cropper.crop(entry) #target size on this takes longer than ffmpeg resize
            cv2.imwrite(path + '\\' + indexstr + ".jpg", result)
        except:
            print("Image could not be processed")

    deletes = glob(r'C:\\Users\\natha\Desktop\\Audviya\\' + subject +r'\\00*.jpg')
    for delete in deletes:
        try:
            os.remove(delete)
        except:
            print("Could not delete")

#def ImageCompare(subject, path):
#    entries = glob(r'C:\\Users\\natha\Desktop\\Audviya\\' + subject +r'\\*.jpg')
#    entries = natsorted(entries)
#    deletes = []
#    for entry in entries[:-1]:
#        if entries.index(entry) < 6:
#            indexstr = str(entries.index(entry))
#            indexstr2 = str(entries.index(entry) + 1)
#            image1 = path + '\\' + indexstr + '.jpg'
#            image2 = path + '\\' + indexstr2 + '.jpg'
#            score = str(round(generateScore(image1, image2), 2))
#            print(f"similarity Score " + indexstr + " to " + indexstr2 + ":", score)
#            if float(score) > 90:
#                deletes.append(path + '\\' + indexstr2 + '.jpg')
#
#    for delete in deletes:
#        os.remove(delete)
#    entries = glob(r'C:\\Users\\natha\Desktop\\Audviya\\' + subject +r'\\*.jpg')
#    entries = natsorted(entries)
#    for entry in entries:
#        indexstr = str(entries.index(entry))
#        os.rename(entry, path + '\\' + indexstr + ".jpg")

   

def BaseAudioImageZoomMargin(sentences, path, subject):
    entries = glob(r'C:\\Users\\natha\Desktop\\Audviya\\' + subject +r'\\*.jpg')
    entries = natsorted(entries)
    #get where everything is coming from #todo (not suppppper important) break the OpenVoice stuff away from it being an all consuming mess
    path_base = r'C:\\Users\\natha\\Desktop\\Audviya\\Scripts\\'
    #use english, will need to change for different languages
    ckpt_base = path_base + 'checkpoints\\base_speakers\\EN'
    #idk tbh for the next two
    ckpt_converter = path_base + 'checkpoints\\converter'
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    ScaleImage = '''[0:v]scale=w=1080:h=-1[ScaleImage]'''
    ScaleVideo = '''[ScaleImage]scale=6000:-1[ScaleVideo]'''
    ZoomPan = '''[ScaleVideo]zoompan=z='zoom+0.0015':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2):d=500:s=1080X1080'[zoom]'''
    Margin = '''[zoom]pad=width=iw:height=ih+2*420:x=0:y=420:color=black'''
    filter_complex = '-filter_complex "' + ScaleImage + ', ' + ScaleVideo + ', ' + ZoomPan + ', ' + Margin + '"' #this might be slower but idk

    #idk what the next 4 things mean and most of what is here until like when I get duration
    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    os.makedirs(path, exist_ok=True)

    source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

    #IMPORTANT must change if I want the speaker to be different
    reference_speaker = r'C:\\Users\\natha\\Desktop\\Audviya\\Scripts\\resources\\NathanShort.wav'
    target_se, audio_name = get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True) #TODO see why audio name not coming out??

    #see for charmap encodes not working with commandline
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    for sentence in sentences:
        indexstr = str(sentences.index(sentence))
        save_path = path + '\\Audio' + indexstr +'.wav'

        # Run the base speaker tts
        text = sentence
        src_path = f'{path}/tmp.wav'
        base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.1)

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
        duration = str(librosa.get_duration(filename=path + '\\Audio' + indexstr +'.wav') + 1) #+ 1 is for padding when doing audio
        #combined image scale, zooming in as well as top border for fitting to TikTok
        command = 'ffmpeg -y -framerate 1 -loop 1 -i '+ indexstr + '.jpg -i Audio' + indexstr + '.wav ' + filter_complex + ' -t ' + duration + ' -pix_fmt yuv420p Video' + indexstr + '.mp4'
        print(command)
        subprocess.call(command,cwd=path, shell=True)

def VideoParts(path):
    LST=glob(path + '\\Audio*.wav')
    print(LST)
    LST = natsorted(LST)
    print(LST)
    durations = []
    index = -1 #setup for end case
    for i in LST:
        indexstr = str(LST.index(i))
        duration = librosa.get_duration(filename=path + '\\Audio' + indexstr +'.wav')
        durations.append(duration)
    print(durations)
    for count in durations:
    # Use slicing to get the first i elements from the list
        index = durations.index(count)
        current_sum = sum(durations[:index]) + 1
        print(current_sum)
        if current_sum > 60 and len(durations[:index]) != len(durations):
            index = index - 1
            print("This is how many videos will be in this video : " + str(index))
            break
        elif current_sum < 60:
            print("This is how many videos will be in this video so far: " + str(index))
    
    #adds the last part when there are is only one more part and the number of videos is greater than 1
    if (len(durations) - 1) == index and len(durations) > 1:
        index = index + 1
    return index
    #Get the number of videos that will be in the first part
    #Create a video on that labeled by the number of times the function was called for "Part X"
    #remove the Videos that were a part of that first one
    #if there is only one video left, do subtitle with Part "X" and end it all
    #if there is more, do the script again from the Video Parts section
    #if there is none, end the script

    
def get_length(filename):
  result=subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of",
                         "default=noprint_wrappers=1:nokey=1", filename],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT)
  return float(result.stdout)

##########################################

def CrossFade(path,subject): #,subject
    #vidslist = glob(path + '\\Video*.mp4')
    #TODO remove and edit with the cwd command work
    ffmpeg_subject = str('"' + subject +'"')
    ffmpeg_path = r'C:\\Users\\natha\\Desktop\\Audviya\\'+ ffmpeg_subject + '\\'

    #combine all of the videos together with space for a crossfade. Very difficult!
    #TODO rename things so that they make more sense
    LST=[]
    DIR= path + '\\'
    for f in os.listdir(DIR):
        if (f.endswith(".mp4")):
            print(f)
            LST.append(f)
    #make all of the video items
    LST = natsorted(LST)
    FLT=""
    #set offset and duration
    OFS=0
    XFD=1
    CNT=0
    XFDA=1

    #make the input different than what is put into ffmpeg because I did not set up the directories well with CWD in ffmpeg call
    INP=[]
    FFMPEG_List=[]
    f=ffmpeg_path+LST[0]
    INP.append(f' -i {f}')
    f=DIR+LST[0]
    FFMPEG_List.append(f' -i {f}')
    PDV='[0:v]'

    #set up video concat with fade for every video and crossfading
    for i in range(len(LST)-1):
        OFS=OFS+get_length(f)-XFD
        FLT=FLT+f'{PDV}'
        CNT=i+1
        PDV=f'[{CNT}v]'
        FLT=FLT+f'[{CNT}:v]xfade=transition=fade:offset={OFS}:duration={XFD}{PDV}'
        if i < len(LST)-2:
            FLT=FLT+";"
        f=ffmpeg_path+LST[i+1]
        INP.append(f' -i {f}')
        INP = (natsorted(INP))
        f=DIR+LST[i+1]
        FFMPEG_List.append(f' -i {f}')
    
    #do everything like b4 but with audio
    PDA = '[0:a]'
    audios = []
    for i in LST:
        index = LST.index(i)
        indexstr = str(LST.index(i))
        CNT = index+1
        next_audio = f'[{CNT}a]'
        FLTA = PDA + f'[{CNT}:a]acrossfade={XFDA}:c1=nofade{next_audio}'
        FLTA=FLTA+";"
        PDA = next_audio
        audios.append(FLTA)
    print(audios)
    audios.pop()
    audios = ''.join(audios)
    audios = audios[:-1]
    #print(audios)

    #remove the last audio bc for some reason an audio gets copied over multiple times
    position_of_character = audios.rfind('[')

    if position_of_character != -1:
        audio_map = audios[position_of_character:]
    else:
        print("Specific character not found in the string.")
    s='ffmpeg'
    for t in INP:
        s=s+t
    INP = (natsorted(INP))

    #combine everything for the filters and the audios
    FLT = FLT + ';' + audios
    print(FLT)

    #write out the filter statment with all the other end stuff
    s=s+f' -filter_complex "{FLT}" -map {PDV} -map {audio_map} -c:v h264_nvenc -cq 18 -c:a aac -q:a 4 -map_metadata -1 -pix_fmt yuv420p -s:v 1080x1920 '+ ffmpeg_path +'VideoUnsub.mp4 -y -hide_banner'
    print(s)
    os.system(s)
    #this is the best thing ever https://www.reddit.com/r/ffmpeg/comments/u3z5y0/cross_fade_arbitrary_number_of_videos_ffmpeg/
    #can maybe apply methodology to create really quick ffmpeg where it does all of the videos as one ffmpeg command


def SubTitle(subject, path, Subtitle_Summary):
    #load whisper through Stable Whisper or stable-ts github group (so useful)
    model = stable_whisper.load_model("base") # or whatever model you prefer

    #align subtitles to original text and not audio
    result = model.align(path + '\\VideoUnsub.mp4', Subtitle_Summary, language='en', suppress_silence=False)
    print(result)
    #stable-whisper result, not sure if best is srt or ASS file tbh
    result.to_srt_vtt(path + '\\VideoFinal.srt', segment_level=False, word_level=True)

    #make it so the subtitle does not appear for 1 second for better thumbnail / cover generation
    subs = pysrt.open(path + '\\VideoFinal.srt')
    subs[0].start.milliseconds = 1
    #first_sub = pysrt.SubRipItem(index=0, start=pysrt.SubRipTime(0, 0, 0, 0), end=pysrt.SubRipTime(0, 0, 0, 1), text="")
    # subs.insert(0, first_sub)
    subs.save(path + '\\VideoFinal.srt')

    #for font path
    font_path = 'NotoSans_Condensed-SemiBold.ttf'
    # Copy the file to the destination directory
    shutil.copy(font_path, path)

    #would like to add as this but it is not working as of right now
    subtitles = '''subtitles=VideoFinal.srt:'''
    force_style = '''force_style=\'Alignment=2,OutlineColour=&H100000000,BorderStyle=3,Outline=1,Shadow=0,Fontsize=20,MarginL=5,MarginV=40\''''
    draw_text = '''drawtext=text=''' + subject + ''':x=(w-text_w)/2:y=290:fontsize='''+TitleFontSize(subject)+''':fontcolor=white:fontfile=NotoSans_Condensed-SemiBold.ttf"'''
    #TODO make the title appear no matter how long it is
    command = ('ffmpeg -y -i VideoUnsub.mp4 -filter_complex "'+ subtitles + force_style +', '+ draw_text +' "'+ subject +'_pre".mp4')
    print(command)
    subprocess.call(command,cwd=path, shell=True)

    #for final path
    final_path = path + '\\' + subject + '_pre.mp4' 
    # Copy the file to the destination directory
    #shutil.copy(final_path, 'C:\\Users\\natha\\Desktop\\Audviya\\V10')

def AddMusic(subject, path):
    #for music path
    music_path = 'C:\\Users\\natha\\Desktop\\Audviya\\Wikipedia-VideoMaker\\Wikipedia-VideoMaker\\music\\Moonlight_Sonata.wav'
    # Copy the file to the destination directory
    shutil.copy(music_path, path)
    duration = str(librosa.get_duration(filename=path + '\\'+ subject + '_pre.mp4'))
    print(duration)
    command = ('ffmpeg -y -i "'+ subject +'_pre".mp4 -i Moonlight_Sonata.wav -filter_complex "[1:a]volume=.5[a2]; [0:a][a2]amix=inputs=2[a]" -t '+ duration +' -map 0:v -map "[a]" -c:v copy "'+ subject +'".mp4')
    subprocess.call(command,cwd=path, shell=True)
    final_path = path + '\\'+ subject +'.mp4'
    shutil.copy(final_path, 'C:\\Users\\natha\\Desktop\\Audviya\\V10')

    return final_path

def DeletePartAll(path, subject): #not necessary it looks like
    LST = []

    LST.append(path + '\\' + 'VideoUnsub.mp4')
    LST.append(path + '\\' + 'VideoFinal.srt')
    LST.append(path + '\\' + subject + '_pre.mp4')
    LST.append(path + '\\' + 'Moonlight_Sonata.mp4')
    LST.append(path + '\\' + subject + '.mp4')

    print(LST)
    for delete in LST:
        try:
            os.remove(delete)
        except:
            continue

################################

## Figure out how to make the audios be the duration instead of the videos
def CrossFadePart(path,subject, index, RanThrough): #,subject
    #TODO remove and edit with the cwd command work
    ffmpeg_subject = str('"' + subject +'"')
    ffmpeg_path = r'C:\\Users\\natha\\Desktop\\Audviya\\'+ ffmpeg_subject + '\\'

    #combine all of the videos together with space for a crossfade. Very difficult!
    #TODO rename things so that they make more sense
    LST = []
    DIR= path + '\\'
    for f in os.listdir(DIR):
        if (f.endswith(".mp4")):
            print(f)
            LST.append(f)

    pattern = r"Video.*\.mp4$"
    LST = [item for item in LST if re.search(pattern, item)]
    print(LST)
    #make all of the video items
    LST = natsorted(LST)
    #only use parts from count
    LST = LST[:index]

    FLT=""
    #set offset and duration
    OFS=0
    XFD=1
    CNT=0
    XFDA=1

    #make the input different than what is put into ffmpeg because I did not set up the directories well with CWD in ffmpeg call
    INP=[]
    FFMPEG_List=[]
    f=ffmpeg_path+LST[0]
    INP.append(f' -i {f}')
    f=DIR+LST[0]
    FFMPEG_List.append(f' -i {f}')
    PDV='[0:v]'

    #set up video concat with fade for every video and crossfading
    for i in range(len(LST)-1):
        OFS=OFS+get_length(f)-XFD
        FLT=FLT+f'{PDV}'
        CNT=i+1
        PDV=f'[{CNT}v]'
        FLT=FLT+f'[{CNT}:v]xfade=transition=fade:offset={OFS}:duration={XFD}{PDV}'
        if i < len(LST)-2:
            FLT=FLT+";"
        ffmpeg_path = ffmpeg_path
        f=ffmpeg_path+LST[i+1]
        INP.append(f' -i {f}')
        INP = (natsorted(INP))
        f=DIR+LST[i+1]
        FFMPEG_List.append(f' -i {f}')
    
    #do everything like b4 but with audio
    PDA = '[0:a]'
    audios = []
    for i in LST:
        index = LST.index(i)
        indexstr = str(LST.index(i))
        CNT = index+1
        next_audio = f'[{CNT}a]'
        FLTA = PDA + f'[{CNT}:a]acrossfade={XFDA}:c1=nofade{next_audio}'
        FLTA=FLTA+";"
        PDA = next_audio
        audios.append(FLTA)
    print(audios)
    audios.pop()
    audios = ''.join(audios)
    audios = audios[:-1]
    #print(audios)

    #remove the last audio bc for some reason an audio gets copied over multiple times
    position_of_character = audios.rfind('[')

    if position_of_character != -1:
        audio_map = audios[position_of_character:]
    else:
        print("Specific character not found in the string.")
    s='ffmpeg'
    for t in INP:
        s=s+t
    INP = (natsorted(INP))

    #combine everything for the filters and the audios
    FLT = FLT + ';' + audios
    print(FLT)

    #write out the filter statment with all the other end stuff
    s=s+f' -filter_complex "{FLT}" -map {PDV} -map {audio_map} -c:v h264_nvenc -cq 18 -c:a aac -q:a 4 -map_metadata -1 -pix_fmt yuv420p -s:v 1080x1920 '+ ffmpeg_path +'VideoUnsubPre'+ str(RanThrough) +'.mp4 -y -hide_banner'
    print(s)
    os.system(s)

    #deletes the last second of the video
    command = 'ffmpeg -i VideoUnsubPre'+ str(RanThrough) +'.mp4 -ss 1 -i VideoUnsubPre'+ str(RanThrough) +'.mp4 -c copy -map 1:0 -map 0 -shortest -f nut - | ffmpeg -f nut -i - -map 0 -map -0:0 -c copy VideoUnsub'+ str(RanThrough) +'.mp4'
    print(command)
    subprocess.call(command,cwd=path, shell=True)
    #this is the best thing ever https://www.reddit.com/r/ffmpeg/comments/u3z5y0/cross_fade_arbitrary_number_of_videos_ffmpeg/
    #can maybe apply methodology to create really quick ffmpeg where it does all of the videos as one ffmpeg command


def SubTitlePart(subject, path, index, RanThrough, sentences):
    #load whisper through Stable Whisper or stable-ts github group (so useful)
    model = stable_whisper.load_model("base") # or whatever model you prefer
    
    print(sentences)
    if len(sentences) > 1:
        sentences_subtitle = sentences[:index]
        del sentences[:index]
        print(sentences)
        sentences_subtitle.insert(0, 'Part' + str(RanThrough) + ' .') #part for first millisecond b4 we actually get into subtitles
        Subtitle_Summary = ''.join(sentences_subtitle)
        print("this is the Subtitle Summary b4 changes: " + Subtitle_Summary)
    else:
        Subtitle_Summary = ''.join(sentences)
        print(Subtitle_Summary)
    #IMPORTANT match this with whatever is happening in the WikipediaGet changes
    summary_pattern = r'[,"():;]'
    #removes the commas, quotations, parenthesis, colon from what is given
    Subtitle_Summary = re.sub(summary_pattern, '', Subtitle_Summary)
    #makes the periods spaces instead of what they were
    summary_pattern = r'[.?!]' #Do not put dashes as it was ruining the script by removing numbers
    Subtitle_Summary = re.sub(summary_pattern, ' ', Subtitle_Summary)

    if index > 0:
        #align subtitles to original text and not audio
        result = model.align(path + '\\VideoUnsub'+ str(RanThrough) +'.mp4', Subtitle_Summary, language='en', suppress_silence=False)
    elif index == 0:
        result = model.align(path + '\\Video0.mp4', Subtitle_Summary, language='en', suppress_silence=False)
    print(result)
    #stable-whisper result, not sure if best is srt or ASS file tbh
    result.to_srt_vtt(path + '\\VideoFinal'+ str(RanThrough) +'.srt', segment_level=False, word_level=True)

    #make it so the subtitle does not appear for 1 second for better thumbnail / cover generation
    subs = pysrt.open(path + '\\VideoFinal'+ str(RanThrough) +'.srt')
    subs[0].start.milliseconds = 0 #test
    subs[0].end.milliseconds = 1 #test
    subs.save(path + '\\VideoFinal'+ str(RanThrough) +'.srt')

    #for font path
    font_path = 'NotoSans_Condensed-SemiBold.ttf'
    # Copy the file to the destination directory
    shutil.copy(font_path, path)

    #would like to add as this but it is not working as of right now #testing with  " Part " + RanThrough + after the subject to see how that looks
    subtitles = 'subtitles=VideoFinal'+ str(RanThrough) +'.srt:'
    force_style_subs = '''force_style=\'Alignment=2,OutlineColour=&H100000000,BorderStyle=3,Outline=1,Shadow=0,Fontsize=17.5,MarginL=5,MarginV=40\''''
    draw_text = '''drawtext=text=''' + subject + ''':x=(w-text_w)/2:y=255:fontsize='''+TitleFontSize(subject)+''':fontcolor=white:fontfile=NotoSans_Condensed-SemiBold.ttf''' #deleted " from end here #moved up the y to fit part below
    draw_text_part = '''drawtext=text=Part '''+ str(RanThrough) +''':x=(w-text_w)/2:y=375:fontsize=40:fontcolor=white:fontfile=NotoSans_Condensed-SemiBold.ttf"''' #testing in top left to see how that looks #todo gotta change from 1
    #TODO make the title appear no matter how long it is
    if index > 0:
        command = ('ffmpeg -y -i VideoUnsub'+ str(RanThrough) +'.mp4 -filter_complex "'+ subtitles + force_style_subs +', '+ draw_text +', '+ draw_text_part +' b4Music'+ str(RanThrough) +'.mp4')
    elif index == 0:
        command = ('ffmpeg -y -i Video0.mp4 -filter_complex "'+ subtitles + force_style_subs +', '+ draw_text +' b4Music'+ str(RanThrough) +'.mp4')
    print(command)
    subprocess.call(command,cwd=path, shell=True)

    return sentences
    #for final path
    # final_path = path + '\\b4Music.mp4' 
    # # Copy the file to the destination directory
    # shutil.copy(final_path, 'C:\\Users\\natha\\Desktop\\Audviya\\V10')

def AddMusicPart(subject, path, RanThrough):
    #for music path
    music_path = 'Moonlight_Sonata.wav'
    # Copy the file to the destination directory
    shutil.copy(music_path, path)
    duration = str(librosa.get_duration(filename=path + '\\b4Music'+ str(RanThrough) +'.mp4'))
    print(duration)
    command = ('ffmpeg -y -i b4Music'+ str(RanThrough) +'.mp4 -i Moonlight_Sonata.wav -filter_complex "[1:a]volume=.5[a2]; [0:a][a2]amix=inputs=2[a]" -t '+ duration +' -map 0:v -map "[a]" -c:v copy "'+ subject +' Part '+ str(RanThrough) +'".mp4')
    subprocess.call(command,cwd=path, shell=True)
    final_path = path + '\\'+ subject +' Part '+ str(RanThrough) +'.mp4'
    shutil.copy(final_path, 'C:\\Users\\natha\\Desktop\\Audviya\\V10')

def DeleteSomePart(path, index):
    LST_video=glob(path + '\\' + 'Video*.mp4')
    #make all of the video items
    LST_video = natsorted(LST_video)
    #only use parts from count
    LST_video = LST_video[:index]

    #delete audio too
    LST_audio=glob(path + '\\' + 'Audio*.wav')
    LST_audio = natsorted(LST_audio)
    #only use parts from count
    LST_audio = LST_audio[:index]

    LST_video.extend(LST_audio)

    LST = LST_video

    print(LST)
    for delete in LST:
        try:
            os.remove(delete)
        except:
            continue

    #rename the video and audio files to random strings
    letter_chars = string.ascii_letters
    LST_audio=glob(path + '\\' + 'Audio*.wav')
    LST_audio=natsorted(LST_audio)
    for i in LST_audio:
        indexstr = str(LST_audio.index(i))
        os.rename(i, path + '\\' + 'Audio'+ indexstr + ''.join(random.choice(letter_chars) for _ in range(15)) +'.wav')

    LST_video=glob(path + '\\' + 'Video*.mp4')
    LST_video=natsorted(LST_video)
    for i in LST_video:
        indexstr = str(LST_video.index(i))
        os.rename(i, path + '\\' + 'Video'+ indexstr + ''.join(random.choice(letter_chars) for _ in range(15)) +'.mp4')

    #rename the video and audio files to correct strings
    LST_audio=glob(path + '\\' + 'Audio*.wav')
    LST_audio=natsorted(LST_audio)
    for i in LST_audio:
        indexstr = str(LST_audio.index(i))
        os.rename(i, path + '\\' + 'Audio'+ indexstr +'.wav')
    
    LST_video=glob(path + '\\' + 'Video*.mp4')
    LST_video=natsorted(LST_video) #TODO remove videounsub
    for i in LST_video:
        indexstr = str(LST_video.index(i))
        os.rename(i, path + '\\' + 'Video'+ indexstr +'.mp4')

def DeleteAll(path):
    #delete all files 
    deletes = glob(path + '\\' + '*')
    for delete in deletes:
        try:
            os.remove(delete)
        except:
            continue
    
    os.rmdir(path)

# ## Figure out how to make the audios be the duration instead of the videos
# def CrossFadeFinal(subject): #,subject
#     #vidslist = glob(path + '\\Video*.mp4')
#     #TODO remove and edit with the cwd command work
#     path = r'C:\\Users\\natha\\Desktop\\Audviya\\V10\\'
#     ffmpeg_subject = str('"' + subject +'"')
#     ffmpeg_path = r'C:\\Users\\natha\\Desktop\\Audviya\\V10\\'

#     #combine all of the videos together with space for a crossfade. Very difficult!
#     #TODO rename things so that they make more sense
#     LST = []
#     DIR= path + '\\'
#     for f in os.listdir(DIR):
#         if subject in f:
#             print(f)
#             LST.append(f)

#     #make all of the video items
#     LST = natsorted(LST)

#     FLT=""
#     #set offset and duration
#     OFS=0
#     XFD=1
#     CNT=0
#     XFDA=1

#     #make the input different than what is put into ffmpeg because I did not set up the directories well with CWD in ffmpeg call
#     INP=[]
#     FFMPEG_List=[]
#     f=ffmpeg_path+LST[0]
#     INP.append(f' -i "{f}"')
#     f=DIR+LST[0]
#     FFMPEG_List.append(f' -i "{f}"')
#     PDV='[0:v]'

#     #set up video concat with fade for every video and crossfading
#     for i in range(len(LST)-1):
#         OFS=OFS+get_length(f)-XFD
#         FLT=FLT+f'{PDV}'
#         CNT=i+1
#         PDV=f'[{CNT}v]'
#         FLT=FLT+f'[{CNT}:v]xfade=transition=fade:offset={OFS}:duration={XFD}{PDV}'
#         if i < len(LST)-2:
#             FLT=FLT+";"
#         ffmpeg_path = ffmpeg_path
#         f=ffmpeg_path+LST[i+1]
#         INP.append(f' -i "{f}"')
#         INP = (natsorted(INP))
#         f=DIR+LST[i+1]
#         FFMPEG_List.append(f' -i "{f}"')
    
#     #do everything like b4 but with audio
#     PDA = '[0:a]'
#     audios = []
#     for i in LST:
#         index = LST.index(i)
#         indexstr = str(LST.index(i))
#         CNT = index+1
#         next_audio = f'[{CNT}a]'
#         FLTA = PDA + f'[{CNT}:a]acrossfade={XFDA}:c1=nofade{next_audio}'
#         FLTA=FLTA+";"
#         PDA = next_audio
#         audios.append(FLTA)
#     print(audios)
#     audios.pop()
#     audios = ''.join(audios)
#     audios = audios[:-1]
#     #print(audios)

#     #remove the last audio bc for some reason an audio gets copied over multiple times
#     position_of_character = audios.rfind('[')

#     if position_of_character != -1:
#         audio_map = audios[position_of_character:]
#     else:
#         print("Specific character not found in the string.")
#     s='ffmpeg'
#     for t in INP:
#         s=s+t
#     INP = (natsorted(INP))

#     #combine everything for the filters and the audios
#     FLT = FLT + ';' + audios
#     print(FLT)

#     #write out the filter statment with all the other end stuff
#     s=s+f' -filter_complex "{FLT}" -map {PDV} -map {audio_map} -c:v h264_nvenc -cq 18 -c:a aac -q:a 4 -map_metadata -1 -pix_fmt yuv420p -s:v 1080x1920 '+ ffmpeg_path + '"' + subject +'".mp4 -y -hide_banner'
#     print(s)
#     os.system(s)
#     #this is the best thing ever https://www.reddit.com/r/ffmpeg/comments/u3z5y0/cross_fade_arbitrary_number_of_videos_ffmpeg/
#     #can maybe apply methodology to create really quick ffmpeg where it does all of the videos as one ffmpeg command


######################################################################################################
    
#Additional functions that are not directly part of flow and are only sometime applicable and may want to delete or change in future
def SummaryChanges(summary):
    lookup_table = {
        'U.S.': 'United States',
        '-': ' ',
        '\u2013':' ',
        '\u2014':' ',
        'I': '1',
        'II': '2',
        'Jr.': 'Junior'
        }
    
    for key, value in lookup_table.items():
        summary = summary.replace(key, value)
    return summary

#change the size of the title based on how long the article title is
def TitleFontSize(subject): #TODO change based on if it is TikTok or YouTube video with lower or not cause then want to make higher
    Length = len(subject)
    if Length <= 18:
        Font_Size = 110
    elif Length > 18 and Length <= 20:
        Font_Size = 100
    elif Length > 20 and Length <= 23:
        Font_Size = 90
    elif Length > 23 and Length <= 27:
        Font_Size = 80
    elif Length > 27 and Length <= 31:
        Font_Size = 70
    else:
        Font_Size=60
    return str(Font_Size)