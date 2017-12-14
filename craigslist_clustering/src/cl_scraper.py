#!usr/bin/python2
#-------------------------------------------------------------------------------------------
# -- SCRIPT NAME: rm_location_scrape.py
# -- LANGUAGE: Python 2.7
# -- PURPOSE: scrape locations information and reviews from rubmaps
#-------------------------------------------------------------------------------------------

from bs4 import BeautifulSoup
import bs4 as bs
import urllib2
import datetime
import mysql.connector
import schedule
from time import sleep

from requests import session
import pandas as pd
import re, os, pickle, time, csv
import sys
import time as t
import random
#import winsound
from itertools import izip
    
# local file
# from rm_cfg import cfg, request_url


# list of browswers as seen by the site, to look like various users, vary to reduce chance of ban
# found samples on https://gist.github.com/seagatesoft/e7de4e3878035726731d
# https://github.com/aivarsk/scrapy-proxies, package that allows you to use proxy ip address
user_agent_list = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
    "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML, like Gecko) Ubuntu/11.04 Chromium/12.0.742.112 Chrome/12.0.742.112 Safari/534.30"
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:23.0) Gecko/20100101 Firefox/23.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.62 Safari/537.36',
    'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; WOW64; Trident/6.0)',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.146 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.146 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64; rv:24.0) Gecko/20140205 Firefox/24.0 Iceweasel/24.3.0',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:28.0) Gecko/20100101 Firefox/28.0',
    'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:28.0) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2',
]


# Save Results in CSV # example filename: 'craigslist_posting_therapeutic.csv'
def save_postings_to_csv(allPostings, fileName, verbose=False):
    if verbose: print('save_postings_to_csv')

    # get file direcotry
    cur_dir = os.path.abspath(os.curdir)
    fn = "{}_withcities.csv".format(fileName)
    file_path = os.path.normpath(os.path.join(cur_dir, "..", "data", fn))

    # write file to disk
    with open(file_path, 'ab') as f:
        writer = csv.writer(f)
        for i in allPostings:
            writer.writerow(str(i.decode('utf-8','ignore')))

def beep():
	pass
    #frequency = 2500  # Set Frequency To 2500 Hertz
    #duration = 1000  # Set Duration To 1000 ms == 1 second
    #winsound.Beep(frequency, duration)

def get_time():
    return t.strftime('%d_%m_%d_%H_%M_%S')

def read_in_city_urls(fn="CraigslistURLs.csv", verbose=True):
    if verbose: print 'Reading csvs'
    cur_dir = os.path.abspath(os.curdir)
    input_dir = os.path.join(cur_dir, "..", "..", "craigslist_clustering", "data" )
    file_path = os.path.normpath(os.path.join(input_dir, fn))
    print(file_path)
    cl_url_df = pd.read_csv(file_path, header=0)
    return cl_url_df["link"].tolist(), cl_url_df["city"].tolist(),

def save_to_database(dataPostings):
    print 'saving to database'
    cnx = mysql.connector.connect(user='shane', password='HT_daen690', host='ec2-52-87-253-53.compute-1.amazonaws.com', database='NOVA_HT_TEST',charset='utf8mb4')
    dbCursor = cnx.cursor()
    for data in allPostings:
        dbCursor.execute('''INSERT INTO CL_EXTRACT (cl_ad_title, cl_url, cl_location, cl_ad, cl_ad_dt, cl_lat, cl_lon, cl_ad_street_addr, date_retrieved, data_source) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', data)
        cnx.commit()
    cnx.close()
    print 'saved_to_database'

def scrape_all_pages(baseURL, city, listing, verbose=False):
    if verbose: print 'scrape_all_pages'
    pageSize = 120
    # scrape the first page of results, and return the total number of result listings
    # as well as the parsed postings.
    (allPostings, totalPostings) = scrape_page(baseURL + "0", city, True, verbose)

    # scrape the remaining result pages. 120 results listed per page by default.
    if totalPostings: # if not None
        i = pageSize
        while i < totalPostings:
            print str(i) + ' postings scraped.'
            try:
                allPostings += scrape_page(baseURL + str(i), city, listing)
            except:
                print("failed to scrape page, saving progress to file")
                save_postings_to_csv(allPostings, 'craigslist_posting_{}'.format(listing))
                sleep(0.5)
                raise
            i += pageSize
        save_postings_to_csv(allPostings, 'craigslist_posting_therapeutic')
    else:
        print("there were no postings")
        pass

    # save_to_database(allPostings)


def scrape_page(therapeuticURL, city, listing, firstPage=False, verbose=False):
    if verbose: print 'scrape_page'
    # Prevent site from 403 Forbidden error
    # get random hdr
    rand_i = random.randrange(0, len(user_agent_list))
    hdr = {'User-Agent':user_agent_list[rand_i]}
    if verbose: print(hdr)

    # queryhttps://github.com/aivarsk/scrapy-proxies
    therapeuticRequest = urllib2.Request(therapeuticURL, headers=hdr)
    baseSoup = BeautifulSoup(urllib2.urlopen(therapeuticRequest), "lxml")
    pagePostings = scrape_postings_from_page(baseSoup, city, listing, verbose)
    if firstPage:
        count_soup = baseSoup.find('span', {'class': ['totalcount']})
        totalPostings = int(count_soup.text) if count_soup else None
        print totalPostings
        return (pagePostings, totalPostings)
    else:
        return pagePostings


# finds and scrapes data from each posting on the current result page
def scrape_postings_from_page(baseSoup, city, listing, verbose=False):
    if verbose: print 'scrape_postings_from_page'
    allPostings = []
    for line in baseSoup.find_all('a', { 'class': ['result-title', 'hdrlink']}):
        postingURL = line.get('href')
        try:
            postingInfo = scrape_posting_information(postingURL, city, verbose)
        except:
            print("scrape_posting_information failed")
            save_postings_to_csv(allPostings, 'craigslist_posting_{}'.format(listing))
            sleep(0.5)
            raise
        allPostings.append(postingInfo)
        sleep(0.5)
    return allPostings

def scrape_posting_information(postingURL, city, verbose=False):
    def get_title(postingSoup):
        postTitle = postingSoup.select('span[id="titletextonly"]')
        return postTitle[0].text.encode('utf-8') if postTitle else ''

    def get_post_body(postingSoup):
        postBody = postingSoup.select('section[id="postingbody"]')
        return postBody[0].text.encode('ascii','ignore').replace('\n', '') if postBody else None

    def get_post_text(postingSoup):
        postBody = get_post_body(postingSoup)
        return re.findall(r'(?<=QR Code Link to This Post).*', postBody) if postBody else ' '

    def get_location_tags(postingSoup):
        locationTags = postingSoup.select('small')
        if len(locationTags) > 0:
            return re.findall(r'[A-z]+', locationTags[0].text.encode('utf-8'))
        else:
            return ''

    def get_post_time(postingSoup):
        # dateAndTimes will be a list of tuples where each tuple contains the capturing groups from our regex
        time_soup = postingSoup.select('time')
        if time_soup:
            dateAndTimes = re.findall(r'datetime=\"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})-\d{4}\"', time_soup[0].encode('utf-8'))
        else:
            dateAndTimes = None
        return map(lambda t: (t[0] + ' ' + t[1]), dateAndTimes) if dateAndTimes else ' '

    def get_lat_long(postingSoup):
        postLatLon = re.findall(r'\-?\d\d\.\d+', str(postingSoup.select('div[id="map"]')))
        # not none and both found
        if postLatLon and len(postLatLon) > 1:
            return postLatLon[0], postLatLon[1] # lat, long
        else:
            return '', ''

    def get_addresses(postingSoup):
        addresses = postingSoup.select('div.mapaddress')
        return addresses[0].decode_contents(formatter="html") if addresses else ''

    if verbose: print 'in scrape_posting'

    # print postingURL
    postingPage = urllib2.urlopen(postingURL)
    postingSoup = BeautifulSoup(postingPage, "lxml")

    # there was not a post title, maybe javascript error?
    postTitle = get_title(postingSoup)

    # if a location is listed on the page, find and save it.
    postLocation = get_location_tags(postingSoup)
    postText = get_post_text(postingSoup)
    postTime = get_post_time(postingSoup)
    postDataSource = 'CL1'

    postDateRetrieved = str(datetime.datetime.now())
    lat, lon = get_lat_long(postingSoup)

    # Parse only address within the div tags
    postStreetAddress = get_addresses(postingSoup)
    return ["", postText[0]+postTitle,
            postText[0], postTitle, postingURL, city, ",".join(postLocation), postTime[0], lat, lon,
            postStreetAddress,postDateRetrieved, postDataSource]

def main():
    verbose = True

    # randomly choose cl listing
    listings = ['thp', 'stp', 'w4w', 'w4m', 'm4m', 'msr', 'cas']
    rand_i = random.randrange(0, len(listings))
    cl_listing = listings[rand_i]

    # read in scrape urls
    cl_city_urls, cities = read_in_city_urls(verbose=verbose)

    url_count = 1
    n_urls = len(cl_city_urls)
    n_runs = 2
    rand_indxs = random.sample(range(n_urls), n_runs)

    # different listing in craigslist like personnals etc.
    # for cl_listing in listings:
    # iterate through the urls in random order
    # hdr = {'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML, like Gecko) \
    #     Ubuntu/11.04 Chromium/12.0.742.112 Chrome/12.0.742.112 Safari/534.30"}
    for indx in rand_indxs:
        if verbose: print("on url {} of {}: {} ".format(url_count, n_urls, cl_city_urls[indx]))
        if verbose: print("this is the list: {} ".format(cl_listing))

        # baseURL = "https://washingtondc.craigslist.org/search/nva/thp?s="
        # url, theraputic message services

        base_url = "{}/search/{}?s=".format(cl_city_urls[indx], cl_listing)

        # postTitle, postingURL, postLocation, time, lat, long, address, dateRetrieved, post_date, ad
        try:
            scrape_all_pages(base_url, cities[indx], cl_listing, verbose)
        except:
            beep()
            raise
        url_count += 1
        sleep(5)

    beep()



    # call the top-level function
    # scrape_all_pages()
    # scheduler
    # schedule.every().monday.do(scrape_all_pages)

if __name__ == "__main__":
    main()
