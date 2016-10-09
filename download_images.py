from bs4 import BeautifulSoup
import requests
import re
import urllib
import urllib2
import os
import sys
import time


def get_image_links(queries):
    """
    Perform series of Bing image searches and saves image links to list. Each
    search only returns 28 results. Use Bing API if you want more control over
    searches.
    """
    images = []

    for query in queries:
        url = 'http://www.bing.com/images/search?q=' + urllib.quote_plus(query) + '&FORM=HDRSC2'
        soup = BeautifulSoup(requests.get(url).text, 'lxml')
        links = [a['src'] for a in soup.find_all('img', {'src': re.compile('mm.bing.net')})]
        images.extend(links)
        time.sleep(5) # wait 5 seconds before next scrape

    return images


def save_images(images, save_dir, image_type):
    """
    Loop through list of image links and save each image to a user-specified
    directory.
    """
    for image in images:
        raw_img = urllib2.urlopen(image).read()
        count = len([i for i in os.listdir(save_dir) if image_type in i]) + 1
        f = open(save_dir + '/' + image_type + '_' + str(count), 'wb')
        f.write(raw_img)
        f.close()


def main():
    """
    User-specified parameters:
    (1) save_dir: directory to save images
    (2) image_type: file name that should be used when saving set of related
        images (e.g. image_type = 'food' results in food_1, food_2, etc.)
    (3) queries: search terms related to image_type (e.g. if image_type = 'food'
        then search terms might be queries = ['tacos', 'pizza', 'pasta'])

    Returns:
    (1) Creates the user-specified save_dir if it doesn't already exist
    (2) Performs Bing image searches using queries and downloads results to save_dir
    """
    save_dir = 'images'
    image_type = 'food'
    queries = ['tacos', 'pizza', 'pasta', 'hamburgers', 'bbq']

    # Make a directory for saving figures if it doesn't already exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = get_image_links(queries)
    save_images(images, save_dir, image_type)


if __name__ == '__main__':
    main()
    sys.exit(0)
