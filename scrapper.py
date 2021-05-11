from bs4 import BeautifulSoup
import csv
import re
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib3.request import RequestMethods
import os

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(html_body):
    texts = html_body.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

def processJobDesc(desc):
    job_desc_text = re.sub('data sci[a-z]+', ' ', desc, re.I)
    job_desc_text = job_desc_text.strip().replace(",","")
    return job_desc_text

def make_soup(url):
    headers : dict = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}
    response = html = None
    for i in range(5):
        response = requests.get(url,  headers = headers)
        if response:
            break
        else:
            time.sleep(2) # wait 2 seconds before another request
    
    if response.status_code == 200:
            html = response.text
    else: #if not response or response == None: 
        print("Scrapping Failed with error, try alternate", response.status_code)
        try:
            html = urllib.request.urlopen(url).read()
            if not html:
                return None # request failed.
        except:
            print("Unknown error while reading..")

    soup = BeautifulSoup(html, "lxml")
    return soup

def scrape(url):
    collected_jobs_ids = set()
    total_count =0
    cities=["Baltimore%2C+MD","Cambridge%2C%20MA", "San+Francisco%2C+CA", "Washington%2C%20DC", "College%20Park%2C%20MD","Redmond%2C%20WA","San%20Carlos%2C%20CA","San+Bernardino%2C+CA","New+Haven,+CT"]
    try:
        for city in cities:
            current_url = url.format(city)
            same_counter = 0
            start = 0
        
            while len(collected_jobs_ids) < 5000 and same_counter < 45:
                driver.get(current_url + f'&start={start}')
                time.sleep(2)
                
                jobs = driver.find_elements_by_class_name('jobsearch-SerpJobCard.unifiedRow.row.result.clickcard')
                for index, job in enumerate(jobs):
                    try:
                        job_id = job.get_attribute('data-jk')

                        if job_id in collected_jobs_ids:
                            same_counter+=1
                            if same_counter >= 45: break
                            continue;
                        
                        same_counter = 0
                        collected_jobs_ids.add(job_id)

                        job_url = job.find_element_by_css_selector('a[data-tn-element="jobTitle"]')
                        

                        soup_data = make_soup(job_url.get_attribute('href'))
                        time.sleep(0.5)
                        
                        try:
                            job_details_html = soup_data.find("div",{"class":"icl-Container--fluid icl-u-xs-p--sm jobsearch-ViewJobLayout-fluidContainer"})
                            if job_details_html == None and job_id in collected_jobs_ids:
                                    collected_jobs_ids.remove(job_id)
                                    continue
                            
                            path = os.path.join(os.getcwd(), 'html')
                            try:
                                os.makedirs(path, exist_ok = True) 
                            except OSError as error: 
                                print("Directory '%s' can not be created" % path)

                            f = open(f'./html/_{total_count}.html','w')
                            total_count+=1
                            f.writelines(str(job_details_html))
                            f.close()
                        
                            jd = job_details_html.find("div",{"class":"jobsearch-JobComponent-description icl-u-xs-mt--md"})
                            job_desc_text = text_from_html(jd)
                            job_desc_text = processJobDesc(job_desc_text)
                            job_title = job_url.get_attribute('title')
                            job_title = re.compile("[^a-zA-Z ]").sub('', job_title)
                            
                            fw = open("CollectedJobs.csv",mode="a", encoding="utf8", newline='')
                            writer = csv.writer(fw, delimiter=",")
                            writer.writerow([job_desc_text, job_title])
                            time.sleep(1)
                        
                        except AttributeError:
                            collected_jobs_ids.remove(job_id)
                            continue
                    
                    except AttributeError or Exception as exec:
                        print(exec)
                        continue
            
                start += 50
                print("Page {0} for {1}\n".format(start/50, city))
                print("Total Collected ",total_count)

    finally:
        fw = open("Collected_Jobs_id.txt",mode="a", encoding="utf8", newline='')
        writer = csv.writer(fw, delimiter=",")
        writer.writerow(collected_jobs_ids)
        print("Scrapping finished")

if __name__ == "__main__":
    driver = webdriver.Chrome('./chromedriver',chrome_options=chrome_options)
    profiles = ["data+science", "software+engineer", "data+engineer"]
    
    for profile in profiles:
        url = "https://www.indeed.com/jobs?q="+profile+"&l={0}&radius=100&limit=50"
        scrape(url)
    driver.close()
    
