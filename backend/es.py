from datetime import datetime
from elasticsearch import Elasticsearch

es = Elasticsearch()

# doc = {
# 'author': 'kimchy',
# 'text': 'Elasticsearch: cool. bonsai cool.', 'timestamp': datetime.now(),
# }
#
# doc1 = {
# 'author': 'rontom1',
# 'text': 'Elasticsearch: cool. bonsai cool.', 'timestamp': datetime.now(),
# }
# res = es.index(index="test-index", doc_type='tweet', id=1, body=doc)
# print(res['created'])
# res = es.get(index="test-index", doc_type='tweet', id=1)
# print(res['_source'])
# es.indices.refresh(index="test-index")
res = es.search(index="militarygeodata", body={"query": {"match_all": {}}, "size": 5000})



print("Got %d Hits:" % res['hits']['total'])
for hit in res['hits']['hits']:
    print(hit)
    doc = {}
    # doc.setdefault(hit['_source']['@timestamp'], datetime.now())
    id= hit['_id']
    doc['地点'] = hit['_source']['地点'].split('c')[0]
    doc['事件描述'] = hit['_source']['事件描述']
    doc['子事件编号']= hit['_source']['子事件编号']
    doc['时间']=str(int(hit['_source']['时间'].split('-')[0])+2) + '-'+ hit['_source']['时间'].split('-')[1]+ '-'+hit['_source']['时间'].split('-')[2]
    # doc['时间']=hit['_source']['时间']
    doc['装备']=hit['_source']['装备']
    doc['数量']=hit['_source']['数量']
    doc['timestamp']: datetime.now()
    doc['方位']=hit['_source']['方位']
    doc['version']=1
    doc['国家或地区']=hit['_source']['国家或地区']
    doc['坐标']=hit['_source']['坐标']
    doc['父事件编号']=hit['_source']['父事件编号']
    doc['类型']=hit['_source']['类型']

    s = es.index(index="militarygeodata", doc_type="data", id=hit['_id'], body=doc)
    print(s['created'])

    #print(hit)
    # print("%(地点)s %(事件描述)s: %(装备)s" % hit["_source"])

# from elasticsearch import Elasticsearch
# client = Elasticsearch()
#
# response = client.search(
#     index="militarygeodata",
#     body={
#       "query": {
#         "filtered": {
#           "query": {
#             "bool": {
#               "must": [{"match": {"地点": "南海"}}],
#               # "must_not": [{"match": {"地点": "越南"}}]
#             }
#           },
#           # "filter": {"term": {"category": "search"}}
#         }
#       },
#       'size': 200
#       # "aggs" : {
#       #   "per_tag": {
#       #     "terms": {"field": "tags"},
#       #     "aggs": {
#       #       "max_lines": {"max": {"field": "lines"}}
#       #     }
#       #   }
#       # }
#     }
# )
#
# print(response)
# for hit in response['hits']['hits']:
#     print(hit['_score'], hit['_source']['地点'])

# for tag in response['aggregations']['per_tag']['buckets']:
#     print(tag['key'], tag['max_lines']['value'])

# from elasticsearch import Elasticsearch
#
# INDEX = 'militarygeodata'
# LOG_HOST = 'http://localhost:9200'
# logserver = Elasticsearch(LOG_HOST)
#
# script = "ctx._source.地点 = 南海难"
#
# update_body = {
#
#     "query": {
#         "bool": {
#             "filter": {
#                 "term": {"地点": "南海"}
#             }
#         }
#     },
#
#     "script": {
#         "source": script,
#         "lang": "painless"
#
#     }
#
# }
#
# update_response = logserver.update_by_query(index=INDEX, body=update_body)

# from elasticsearch import Elasticsearch
# client = Elasticsearch()
#
# response = client.update_by_query(
#     index="militarygeodata",
#     body={
#         "query": {
#             "bool": {
#                 "must": [{"match": {"地点": "南沙群岛"}}],
#                 "must_not": [{"match": {"description": "海军"}}]
#             }
#         },
#         "script":{
#             "source": "ctx._source.likes++",
#             "lang": "painless"
#         }
#     },
#   )