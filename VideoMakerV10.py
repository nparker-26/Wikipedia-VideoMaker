from mods import *
import FunctionsV10


start_time = time.time()


# Check if command line arguments were passed
if len(sys.argv) > 2:
    email = sys.argv[1] #email is not passed in quotations
    subjects = sys.argv[2:] #subject must be passed in quotations
else:
    print("Please provide a subject and at least one email address as command line arguments.")
    sys.exit(1)

for subject in subjects:

    path = 'C:\\Users\\natha\\Desktop\\Audviya\\'+ subject
    
    #Wikipedia
    summary, sentences, ImageCount = FunctionsV10.WikipediaSummaryGet(subject)
    print(len(sentences))

    #URL summary
    #summary, sentences, ImageCount = FunctionsV7.URLSummaryGet(subject, path)

    end_time = time.time()
    print(f"The summary script took {end_time - start_time} seconds to run.")

    #make the first sentence not have so many of the bad things in it
    first = FunctionsV10.ImproveSentences(sentences)
    sentences[0] = first
    # print(sentences)

    end_time = time.time()
    print(f"Improving sentences took {end_time - start_time} seconds to run.")
    #print(Updated_Summary)

    #Get images
    bing_crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': path})
    bing_crawler.crawl(keyword=subject, offset=0, max_num=ImageCount)

    end_time = time.time()
    print(f"The Image Getting script took {end_time - start_time} seconds to run.")

    #Crop and resize images
    FunctionsV10.CropImage(subject, path)

    end_time = time.time()
    print(f"The cropping script took {end_time - start_time} seconds to run.")

    #make sure no images next to each other are the same
    #FunctionsV10.ImageCompare(subject, path)

    end_time = time.time()
    print(f"The Image Compare took {end_time - start_time} seconds to run.")

    #Make the base video and audio with the video zooming in and having a margin too
    FunctionsV10.BaseAudioImageZoomMargin(sentences, path, subject)

    end_time = time.time()
    print(f"The Base Video script took {end_time - start_time} seconds to run.")

    #Make the videos have a crossfade effect and combined together
    FunctionsV10.CrossFade(path, subject) #, subject

    end_time = time.time()
    print(f"The Cross Fade script took {end_time - start_time} seconds to run.")

    Subtitle_Summary = ''.join(sentences)
    #IMPORTANT match this with whatever is happening in the WikipediaGet changes
    summary_pattern = r'[,"():;]'
    #removes the commas, quotations, parenthesis, colon from what is given
    Subtitle_Summary = re.sub(summary_pattern, '', Subtitle_Summary)
    #makes the periods spaces instead of what they were
    summary_pattern = r'[.]'
    Subtitle_Summary = re.sub(summary_pattern, ' ', Subtitle_Summary)

    #do tts on the video, add the title too
    FunctionsV10.SubTitle(subject, path, Subtitle_Summary)

    end_time = time.time()
    print(f"The total script took {end_time - start_time} seconds to run.")
    
    #Music is not part of this iteration
    #final_path = FunctionsV10.AddMusic(subject, path)
    
    #TODO change email to something in Azure DevOps
    sender_email = 'nathansparker26@gmail.com'
    sender_password = 'xiba nczu ibvz yeav' #this is an app password
    receiver_email = email

    FunctionsV10.send_email(sender_email, sender_password, receiver_email, subject, path)

    # #try adding delete to remove problems #This is bad
    # FunctionsV10.DeletePartAll(path, subject)
    '''
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

    if librosa.get_duration(filename=final_path) > 60: #r'C:\\Users\\natha\\Desktop\\Audviya\\V10\\' + subject + '.mp4'
        RunThrough(RanThrough, sentences)
    else:
        FunctionsV10.DeleteAll(path)
    '''
    FunctionsV10.DeleteAll(path)

    end_time = time.time()
    print(f"The total script took {end_time - start_time} seconds to run.")