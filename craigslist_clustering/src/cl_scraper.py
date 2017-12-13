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

# local file
from rm_cfg import cfg, request_url


def read_in_city_urls(fn="CraigslistURLs.csv", verbose=False):
    if verbose: print 'Reading csvs'
    cur_dir = os.path.abspath(os.curdir)
    input_dir = os.path.join(cur_dir, "..", "..","inputs" )
    file_path = os.path.join(input_dir, fn)
    cl_url_df = pd.read_csv(file_path, header=0)
    return cl_url_df["link"].tolist()

def save_to_database(dataPostings):
    print 'saving to database'
    cnx = mysql.connector.connect(user='melissa', password='HT_daen690', host='ec2-52-87-253-53.compute-1.amazonaws.com', database='NOVA_HT_TEST',charset='utf8mb4')
    dbCursor = cnx.cursor()
    for data in allPostings:
        dbCursor.execute('''INSERT INTO CL_EXTRACT (cl_ad_title, cl_url, cl_location, cl_ad, cl_ad_dt, cl_lat, cl_lon, cl_ad_street_addr, date_retrieved, data_source) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', data)
        cnx.commit()
    cnx.close()
    print 'saved_to_database'

def scrape_all_pages(baseURL, verbose=False):
    if verbose: print 'scrape_all_pages'
    pageSize = 120
    # scrape the first page of results, and return the total number of result listings
    # as well as the parsed postings.
    (allPostings, totalPostings) = scrape_page(baseURL + "0", True, verbose)

    # scrape the remaining result pages. 120 results listed per page by default.
    if totalPostings: # if not None
        i = pageSize
        while i < totalPostings:
            print str(i) + ' postings scraped.'
            allPostings += scrape_page(baseURL + str(i))
            i += pageSize
        save_postings_to_csv(allPostings, 'craigslist_posting_therapeutic.csv')
    else:
        print("there were no postings")
        pass

    # save_to_database(allPostings)


def scrape_page(therapeuticURL, firstPage=False, verbose=False):
    if verbose: print 'scrape_page'
    # Prevent site from 403 Forbidden error
    hdr = {'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML, like Gecko) \
    Ubuntu/11.04 Chromium/12.0.742.112 Chrome/12.0.742.112 Safari/534.30"}
    therapeuticRequest = urllib2.Request(therapeuticURL, headers=hdr)
    baseSoup = BeautifulSoup(urllib2.urlopen(therapeuticRequest), "lxml")
    pagePostings = scrape_postings_from_page(baseSoup, verbose)
    if firstPage:
        count_soup = baseSoup.find('span', {'class': ['totalcount']})
        totalPostings = int(count_soup.text) if count_soup else None
        print totalPostings
        return (pagePostings, totalPostings)
    else:
        return pagePostings


# finds and scrapes data from each posting on the current result page
def scrape_postings_from_page(baseSoup, verbose=False):
    if verbose: print 'scrape_postings_from_page'
    allPostings = []
    for line in baseSoup.find_all('a', { 'class': ['result-title', 'hdrlink']}):
        postingURL = line.get('href')
        postingInfo = scrape_posting_information(postingURL, verbose)
        allPostings.append(postingInfo)
        sleep(0.5)
    return allPostings

def scrape_posting_information(postingURL, verbose=False):
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

    return [postTitle, postingURL, postLocation, ','.join(postText[0], postTime[0], lat, lon, postStreetAddress, postDateRetrieved, postDataSource]

# Save Results in CSV # example filename: 'craigslist_posting_therapeutic.csv'
def save_postings_to_csv(allPostings, fileName, verbose=False):
    if verbose: print 'save_postings_to_csv'
    with open(fileName, 'wb') as f:
        writer = csv.writer(f)
        for i in allPostings:
            writer.writerow(i)
def main():
    verbose = True
    listings = ['stp', 'w4w', 'w4m', 'm4m', 'msr', 'cas']
    cl_city_urls = read_in_city_urls(verbose=verbose)

    url_count = 1
    n_urls = len(cl_city_urls)

    # different listing in craigslist like personnals etc.
    # for cl_listing in listings:
    for url in cl_city_urls:
        if verbose: print("on url {} of {}: {} ".format(url_count, n_urls, url))
        # baseURL = "https://washingtondc.craigslist.org/search/nva/thp?s="
        # url, sublocation, theraputic message services
        base_url = "{}/search/{}?s=".format(url, "thp")
        scrape_all_pages(base_url, verbose)
        url_count += 1


    # call the top-level function
    # scrape_all_pages()
    # scheduler
    # schedule.every().monday.do(scrape_all_pages)

if __name__ == "__main__":
    main()
