#https://github.com/codelucas/newspaper
import newspaper
from tqdm import tqdm
import pandas as pd

#outputPath
output_path = "/CollectAndProcessData/Data/articles.csv"

#classes
class Media():
     def __init__(self, name="", url=""):
          self.name = name
          self.url = url
          self.articles = []
          self.n_articles = 0
          self.download_failed_count = 0
          self.bias = "Right" if name == "Fox News" else "Left" #TODO: make it general.
     def find_articles(self):
          media_paper = newspaper.build(self.url, memoize_articles=False)
          for article in tqdm(media_paper.articles):
               try:
                    newspaper_article = Article(article)
                    self.articles.append(newspaper_article)
                    self.n_articles += 1
               except:
                    self.download_failed_count += 1

class Article():
     def __init__(self, newspaper_article):
          newspaper_article.download()
          newspaper_article.parse()

          self.url = newspaper_article.url
          self.html = newspaper_article.html
          self.authors = newspaper_article.authors
          self.publish_date = newspaper_article.publish_date
          self.text = newspaper_article.text
          self.title = newspaper_article.title
     def __str__(self):
          return str({"url": self.url, "title": self.title, "text": self.text})
     def save_str(self):
          return "" #TODO: return a json file to print. Take inot account "  ' and other characters


def simple_text(text):
     if text is None:
          return ""
     else:
          return text.replace(";",",").replace("\n","").replace("'","").replace('"',"")
 
#define the medias to download
medias = [
     Media("NYTimes", "https://www.nytimes.com"), # Left-center, factual reporting High https://mediabiasfactcheck.com/new-york-times/
     Media("Fox News", "https://www.foxnews.com/"), # Extrem Right, factual reporting Mixed https://mediabiasfactcheck.com/fox-news-bias/
     ]

print("Start collecting articles")
for media in medias:
     media.find_articles()
print("Finished collecting articles")

print("Start saving articles")
n=0
output_df = pd.DataFrame(columns=['media_name', 'media_url', 'article_url', 'article_title', 'article_text' ])
for media in medias:
     for article in media.articles:
          row_df = pd.DataFrame({
               'media_name': [media.name], 
               'media_url': [media.url], 
               'media_bias': [media.bias],
               'article_url': [article.url], 
               'article_title': [simple_text(article.title)], 
               'article_text': [simple_text(article.text)],
               })
          output_df = pd.concat([output_df, row_df], ignore_index=True)
          n+=1
output_df.to_csv(output_path, sep=";") #To read use: >>> data = pd.read_csv("articles.csv", sep=";")
print("Finished saving articles. Number of articles:", n)
