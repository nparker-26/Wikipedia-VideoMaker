from mods import *

base_path = '/'

def WikipediaSummaryGet(url):
    article = Article(url)
    article.download()
    article.parse()
    subject = article.title
    summary = wikipedia.summary(subject, auto_suggest=False)
    summary = SummaryChanges(summary)
    print(subject)

    # Split the paragraph into sentences
    summary = re.sub(r"\[.*?\]|\\(.*?\\)", "", summary)  # Removes brackets from article
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(summary)

    # Count the number of sentences to produce a certain number of images
    ImageCount = max(int(len(sentences) * 1.25), 10)
    return subject, sentences, ImageCount

def URLSummaryGet(subject, path, url):
    full_path = path
    os.makedirs(full_path, exist_ok=True)
    article = Article(url)
    article.download()
    article.parse()
    with open(os.path.join(full_path, f'{subject}.txt'), "w", encoding="utf-8") as file:
        article_text = article.text
        file.write(article_text)
    
    with open(os.path.join(full_path, f'{subject}.txt'), "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Clean the result obtained from the article parser
    lines = [line for line in lines if line.strip()]
    lines = [re.sub(r'\[[^]]*\]', '', line) for line in lines]  # Remove everything in brackets for Wikipedia references
    lines = [line.strip() for line in lines if line.endswith(('.', '?', '!'))]

    for line in lines:
        print(line)
    text = " ".join(lines)

    # Facebook BART text generation
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', model_max_length=512)
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')

    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=125, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print('-------------------------------------')
    print(summary)

    # Split the paragraph into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(summary)

    # Count the number of sentences to produce a certain number of images
    ImageCount = max(int(len(sentences) * 1.25), 10)
    return summary, sentences, ImageCount

def ImproveSentences(sentences):
    first = sentences[0]
    try:
        first = re.sub(r"\(.*?\)", "", first)  # Remove the parentheses and the content inside them
        first = re.sub(' +', ' ', first).strip()
        print(first)
        return first
    except Exception as e:
        print(f"Error: {e}")

def CropImage(subject, path):
    cropper = Cropper()
    entries = glob(os.path.join(path, '*.jpg'))
    entries = natsorted(entries)
    for entry in entries:
        try:
            indexstr = str(entries.index(entry))
            result = cropper.crop(entry)  # Cropping to focus on important content
            cv2.imwrite(os.path.join(path, f'{indexstr}.jpg'), result)
        except Exception as e:
            print(e)
            print("Image could not be processed")

    # Delete temporary images
    deletes = glob(os.path.join(path, '00*.jpg'))
    for delete in deletes:
        try:
            os.remove(delete)
        except Exception as e:
            print(e)
            print("Could not delete")

def BaseAudioImageZoomMargin(sentences, path, subject):
    entries = glob(os.path.join(base_path, subject, '*.jpg'))
    entries = natsorted(entries)
    ckpt_base = os.path.join(base_path, 'checkpoints', 'base_speakers', 'EN')
    ckpt_converter = os.path.join(base_path, 'checkpoints', 'converter')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ScaleImage = '''[0:v]scale=w=480:h=-1[ScaleImage]'''
    ScaleVideo = '''[ScaleImage]scale=6000:-1[ScaleVideo]'''
    ZoomPan = '''[ScaleVideo]zoompan=z='zoom+0.0010':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2):d=500:s=480X480'[zoom]'''
    Margin = '''[zoom]pad=width=iw:height=ih+2*187:x=0:y=187:color=black'''
    filter_complex = '-filter_complex "' + ScaleImage + ', ' + ScaleVideo + ', ' + ZoomPan + ', ' + Margin + '"' #this might be slower but idk
    print(filter_complex)

    base_speaker_tts = BaseSpeakerTTS(os.path.join(ckpt_base, 'config.json'), device=device)
    base_speaker_tts.load_ckpt(os.path.join(ckpt_base, 'checkpoint.pth'))

    tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)
    tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))

    os.makedirs(path, exist_ok=True)

    source_se = torch.load(os.path.join(ckpt_base, 'en_default_se.pth')).to(device)
    reference_speaker = os.path.join(base_path, 'resources', 'NathanShort.wav')
    target_se, audio_name = get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

    for sentence in sentences:
        indexstr = str(sentences.index(sentence))
        save_path = os.path.join(path, f'Audio{indexstr}.wav')
        src_path = os.path.join(path, 'tmp.wav')
        base_speaker_tts.tts(text=sentence, output_path=src_path, speaker='default', language='English', speed=1.1)

        tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se, output_path=save_path, message="@MyShell")
        duration = str(librosa.get_duration(path=save_path) + 1)  # +1 for padding
        command = 'ffmpeg -y -framerate 1 -loop 1 -i '+ indexstr + '.jpg -i Audio' + indexstr + '.wav ' + filter_complex + ' -t ' + duration + ' -pix_fmt yuv420p Video' + indexstr + '.mp4'
        print(command)
        subprocess.call(command, cwd=path, shell=True)

def VideoParts(path):
    full_path = path
    LST=glob(os.path.join(full_path, 'Audio*.wav'))
    print(LST)
    LST = natsorted(LST)
    print(LST)
    durations = []
    index = -1 #setup for end case
    for i in LST:
        indexstr = str(LST.index(i))
        duration = librosa.get_duration(filename=os.path.join(full_path, 'Audio' + indexstr +'.wav'))
        durations.append(duration)
    print(durations)
    for count in durations:
        index = durations.index(count)
        current_sum = sum(durations[:index]) + 1
        print(current_sum)
        if current_sum > 60 and len(durations[:index]) != len(durations):
            index = index - 1
            print("This is how many videos will be in this video : " + str(index))
            break
        elif current_sum < 60:
            print("This is how many videos will be in this video so far: " + str(index))
    
    if (len(durations) - 1) == index and len(durations) > 1:
        index = index + 1
    return index

def get_length(filename):
  result=subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of",
                         "default=noprint_wrappers=1:nokey=1", filename],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT)
  return float(result.stdout)

def CrossFade(path,subject):
    ffmpeg_subject = str('"' + subject +'"')
    ffmpeg_path = '/data/'+ ffmpeg_subject +'/'

    LST=[]
    DIR= path + '/'
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

    #write out the filter statment with all the other end stuff # TODO make hevc_nvenc work (issue using CUDA effectively)
    s=s+f' -filter_complex "{FLT}" -map {PDV} -map {audio_map} -c:v libx265 -cq 18 -c:a aac -q:a 4 -map_metadata -1 -pix_fmt yuv420p -s:v 480x854 '+ ffmpeg_path +'VideoUnsub.mp4 -y -hide_banner'
    # s=s+f' -filter_complex "{FLT}" -map {PDV} -map {audio_map} -c:v libx265 -cq 18 -c:a aac -q:a 4 -map_metadata -1 -pix_fmt yuv420p -s:v 1080x1920 '+ ffmpeg_path +'VideoUnsub.mp4 -y -hide_banner'
    print(s)
    os.system(s)

def SubTitle(subject, path, Subtitle_Summary):
    model = stable_whisper.load_model("base") #TODO run in dockerfile
    full_path = path
    result = model.align(os.path.join(full_path, 'VideoUnsub.mp4'), Subtitle_Summary, language='en', suppress_silence=False)
    print(result)
    result.to_srt_vtt(os.path.join(full_path, 'VideoFinal.srt'), segment_level=False, word_level=True)

    subs = pysrt.open(os.path.join(full_path, 'VideoFinal.srt'))
    subs[0].start.milliseconds = 1
    subs.save(os.path.join(full_path, 'VideoFinal.srt'))

    font_path = 'NotoSans_Condensed-SemiBold.ttf'
    shutil.copy(font_path, full_path)

    subtitles = '''subtitles=VideoFinal.srt:'''
    force_style = '''force_style=\'Alignment=2,OutlineColour=&H100000000,BorderStyle=3,Outline=1,Shadow=0,Fontsize=20,MarginL=5,MarginV=40\''''
    draw_text = '''drawtext=text=''' + subject + ''':x=(w-text_w)/2:y=125:fontsize='''+TitleFontSize(subject)+''':fontcolor=white:fontfile=NotoSans_Condensed-SemiBold.ttf"'''
    command = ('ffmpeg -y -i VideoUnsub.mp4 -filter_complex "'+ subtitles + force_style +', '+ draw_text +' "'+ subject +'".mp4')
    print(command)
    subprocess.call(command,cwd=path, shell=True)

def send_email(sender_email, sender_password, receiver_email, subject, path):
    
    video_path = path + '/' + subject + '.mp4'
    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Wikipedia Video: ' + subject

    # Attach the video file
    with open(video_path, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename= ' + subject + '.mp4')
    msg.attach(part)

    # Connect to the SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    app_pass = 'xiba nczu ibvz yeav'

    # Send the email
    server.send_message(msg)
    server.quit()

def AddMusic(subject, path):
    music_path = os.path.join(base_path, 'Wikipedia-VideoMaker', 'Wikipedia-VideoMaker', 'music', 'Moonlight_Sonata.wav')
    full_path = path
    shutil.copy(music_path, full_path)
    duration = str(librosa.get_duration(filename=os.path.join(full_path, subject + '_pre.mp4')))
    print(duration)
    command = ('ffmpeg -y -i "'+ subject +'_pre".mp4 -i Moonlight_Sonata.wav -filter_complex "[1:a]volume=.5[a2]; [0:a][a2]amix=inputs=2[a]" -t '+ duration +' -map 0:v -map "[a]" -c:v copy "'+ subject +'".mp4')
    subprocess.call(command,cwd=full_path, shell=True)
    final_path = os.path.join(full_path, subject +'.mp4')
    shutil.copy(final_path, os.path.join(base_path, 'V10'))

    return final_path

def DeletePartAll(path, subject):
    LST = []
    full_path = path
    LST.append(os.path.join(full_path, 'VideoUnsub.mp4'))
    LST.append(os.path.join(full_path, 'VideoFinal.srt'))
    LST.append(os.path.join(full_path, subject + '_pre.mp4'))
    LST.append(os.path.join(full_path, 'Moonlight_Sonata.mp4'))
    LST.append(os.path.join(full_path, subject + '.mp4'))

    print(LST)
    for delete in LST:
        try:
            os.remove(delete)
        except:
            continue

def DeleteAll(path):
    full_path = path
    deletes = glob(os.path.join(full_path, '*'))
    for delete in deletes:
        try:
            os.remove(delete)
        except:
            continue
    
    os.rmdir(full_path)

def SummaryChanges(summary):
    lookup_table = {
        'U.S.': 'United States',
        '-': ' ',
        '\u2013':' ',
        '\u2014':' ',
        'II': '2',
        'Jr.': 'Junior'
        }
    
    for key, value in lookup_table.items():
        summary = summary.replace(key, value)
    return summary

def TitleFontSize(subject):
    Length = len(subject)
    if Length <= 18:
        Font_Size = 55
    elif Length > 18 and Length <= 20:
        Font_Size = 50
    elif Length > 20 and Length <= 23:
        Font_Size = 45
    elif Length > 23 and Length <= 27:
        Font_Size = 40
    elif Length > 27 and Length <= 31:
        Font_Size = 35
    else:
        Font_Size=30
    return str(Font_Size)



