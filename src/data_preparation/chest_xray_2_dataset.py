#==============================================================================
# Author       : Abbas R. Ali
# Last modified: Novemeber 20, 2017
# Description  : prepare x-ray dataset
#==============================================================================

from lxml import html
import requests
import re
import json
import urllib
import sys
# import os
# import logging
from src.utils import getConfig, checkPathExists

final_data = {}
img_no = 0
# testset_proportion = getConfig['testset_proportion']

def download_data(url, domain, corpus_dir, dataset):
    try:
        global img_no

        try :
            img_no += 1
            r = requests.get(url)
            tree = html.fromstring(r.text)

            div = tree.xpath('//table[@class="masterresultstable"]//div[@class="meshtext-wrapper-left"]')
        except : div=[]

        if div != []:
            div = div[0]
        else:
            return

        typ = div.xpath('.//strong/text()')[0]
        items = div.xpath('.//li/text()')
        img = tree.xpath('//img[@id="theImage"]/@src')[0]

        final_data[img_no] = {}
        final_data[img_no]['type'] = typ
        final_data[img_no]['items'] = items
        final_data[img_no]['img'] = domain + img
        try:
            urllib.urlretrieve(domain+img, corpus_dir + str(img_no) + ".png")
            with open('data_new.json', 'w') as f:
                json.dump(final_data, f)

            output = "Downloading Images : {}".format(img_no)
            sys.stdout.write("\r\x1b[K" + output)
            sys.stdout.flush()
        except: return
    except Exception as e:
        print(dataset + " data download failed - " + str(e))

def xray_data_extraction(corpus_dir, dataset):
    try:
        print("Downloading " + dataset + " data...")

        checkPathExists([corpus_dir])

        domain = 'https://openi.nlm.nih.gov/'
        url_list = []
        for i in range(0, 75):
            url = 'https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m=' + str(1 + 100 * i) + '&n=' + str(100 + 100 * i)
            url_list.append(url)
        regex = re.compile(r"var oi = (.*);")

        for url in url_list:
            try:
                r = requests.get(url)
            except : continue
            tree = html.fromstring(r.text)

            script = tree.xpath('//script[@language="javascript"]/text()')[0]

            json_string = regex.findall(script)[0]
            json_data = json.loads(json_string)

            # next_page_url = tree.xpath('//footer/a/@href')

            links = [domain + x['nodeRef'] for x in json_data]
            for link in links:
                download_data(link, domain, corpus_dir, dataset)

        print(dataset + " data downloaded successfully")
    except Exception as e:
        print(dataset + " data extraction failed - " + str(e))

# main function
def main():
    try:
        main_dir = '../'

        gConfig = getConfig(main_dir + 'config/metavision.ini')  # get configuration

        dataset = gConfig['datasets']
        corpus_dir = main_dir + gConfig['corpus_dir'] + "/"
        data_dir = main_dir + gConfig['data_dir'] + "/"

        xray_data_extraction(corpus_dir + dataset + "/", dataset)

        # xray_data_preparation(corpus_dir + dataset + "/", data_dir + dataset + "/", dataset)

    except Exception as ex:
        print("main function failed - " + str(ex))
        raise ex

# main function
if __name__ == '__main__':
    main()