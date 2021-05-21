import scrapy, sqlite3, json, threading


class UnsplashSpider(scrapy.Spider):
    def start_requests(self):
        createDB()
        pre = "https://api.unsplash.com/search/photos/?client_id="
        key = ""  # YOUR_ACCESS_KEY
        search = "&query="
        aft, lst = "&page=", "&per_page=30"
        begin, page = 1, 1000
        conn = sqlite3.connect("./database/link.db")
        semaphore = threading.Semaphore(1)
        list_key = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                    'tvmonitor'}
        for search_key in list(list_key):
            for i in range(begin, begin + page + 1):
                yield scrapy.Request(url=pre + key + search + search_key + aft + str(i) + lst,
                                     callback=lambda response, conn=conn, semaphore=semaphore: self.toDB(response, conn,
                                                                                                         semaphore,
                                                                                                         search_key))

    def parse(self, response):
        pass

    def toDB(self, response, conn, semaphore, search_key):
        js = json.loads(response.body_as_unicode())
        for j in js["results"]:
            link = j["urls"]["raw"]
            sql = "INSERT INTO LINK(LINK,CLASS) VALUES ('%s','%s');" % (link, search_key)
            conn.execute(sql)
        semaphore.acquire()
        conn.commit()
        semaphore.release()


def createDB():
    conn = sqlite3.connect("./database/link.db")
    conn.execute("DROP TABLE IF EXISTS LINK;")
    conn.execute("CREATE TABLE LINK ("
                 "ID INTEGER PRIMARY KEY AUTOINCREMENT,"
                 "LINK VARCHAR(255),"
                 "CLASS VARCHAR(255));")
