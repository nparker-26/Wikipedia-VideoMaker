from mods import *
import FunctionsV10

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    email = request.json['email']
    subject = request.json['subject']
    start_time = time.time()
    
    path = '/data/' + subject

    summary, sentences, ImageCount = FunctionsV10.WikipediaSummaryGet(subject)

    first = FunctionsV10.ImproveSentences(sentences)
    sentences[0] = first

    bing_crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': path})
    bing_crawler.crawl(keyword=subject, offset=0, max_num=ImageCount)

    FunctionsV10.CropImage(subject, path)

    FunctionsV10.BaseAudioImageZoomMargin(sentences, path, subject)

    FunctionsV10.CrossFade(path, subject)

    Subtitle_Summary = ''.join(sentences)
    summary_pattern = r'[,"():;]'
    Subtitle_Summary = re.sub(summary_pattern, '', Subtitle_Summary)
    summary_pattern = r'[.]'
    Subtitle_Summary = re.sub(summary_pattern, ' ', Subtitle_Summary)

    FunctionsV10.SubTitle(subject, path, Subtitle_Summary)

    sender_email = 'nathansparker26@gmail.com'
    sender_password = 'xiba nczu ibvz yeav' #this is an app password
    receiver_email = email

    FunctionsV10.send_email(sender_email, sender_password, receiver_email, subject, path)

    FunctionsV10.DeleteAll(path)

    ################ For future use potentially ##############
    '''
    final_path = FunctionsV10.AddMusic(subject, path)

    RanThrough = 0

    def RunThrough(RanThrough, sentences):
        index = FunctionsV10.VideoParts(path)

        if index == 0:
            RanThrough = RanThrough + 1
            sentences = FunctionsV10.SubTitlePart(subject, path, index, RanThrough, sentences)
            FunctionsV10.AddMusicPart(subject, path, RanThrough)
            FunctionsV10.DeleteAll(path)
        elif index > 0:
            RanThrough = RanThrough + 1
            FunctionsV10.CrossFadePart(path, subject, index, RanThrough)
            sentences = FunctionsV10.SubTitlePart(subject, path, index, RanThrough, sentences)
            FunctionsV10.AddMusicPart(subject, path, RanThrough)
            FunctionsV10.DeleteSomePart(path, index)
            RunThrough(RanThrough, sentences)
        else:
            print("index is " + str(index))
            FunctionsV10.DeleteAll(path)
            print(index)

    if librosa.get_duration(filename=final_path) > 60:
        RunThrough(RanThrough, sentences)
    else:
        FunctionsV10.DeleteAll(path)

    '''
    end_time = time.time()
    return {"message": f"The total script took {end_time - start_time} seconds to run."}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)

