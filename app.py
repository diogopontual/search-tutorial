import re
from flask import Flask, render_template, request
from search import Search

app = Flask(__name__)
es = Search()

def extract_filters(query):
    filters = []
    filter_regex = r'category:([^\s]+)\s*'
    m = re.search(filter_regex, query)
    
    if m: 
        filters.append(
            {
                'term': {
                    'category.keyword': {
                        'value': m.group(1)
                    }
                }
        })
        query = re.sub(filter_regex, '', query).strip
    
    filter_regex = r'year:([0-9]{4})\s*'
    m = re.search(filter_regex, query)        
    
    if m: 
        filters.append({
            'range': {
                'updated_at': {
                    'gte': f'{m.group(1)}||/y',
                    'lte': f'{m.group(1)}||/y',
                }
            }           
        })
        query = re.sub(filter_regex, '', query).strip()
    
    return {'filter': filters}, query
    
@app.get('/')
def index():
    return render_template('index.html')


@app.post('/')
def handle_search():
    query = request.form.get('query', '')
    filters, parsed_query = extract_filters(query)
    from_ = request.form.get('from_',type=int,default=0)
    if parsed_query:
        search_query = {
            'must': [{
                'multi_match': {
                    'query': query,
                    'fields': ['name','summary','content']
                }
            }]
        }
    else:
        search_query = {
            'match_all': {}
        }
        
        
    # results = es.search(
    #     query={
    #         'bool': {
    #             ,
    #             **filters
    #         }            
    #     },
    #     size=5,from_=from_
    # )
    results = es.search(
        query={
            'bool': {
                **search_query,
                **filters
            } 
        },
        knn={
            'field': 'embedding',
            'query_vector': es.get_embedding(parsed_query),
            'num_candidates': 50,
            'k': 10
        
        },
        rank={
            'rff':{}
        },
        aggs={
            'category-agg':{
                'terms':{
                    'field':'category.keyword'
                }
            },
            'year-agg':{
                'date_histogram':{
                    'field':'updated_at',
                    'calendar_interval':'year',
                    'format':'yyyy'
                }
            }
        },
        size=5,
        from_=from_
    )
    
    aggs = {
        'Category': {
            bucket['key']: bucket['doc_count']
            for bucket in results['aggregations']['category-agg']['buckets']
        },
        'Year': {
            bucket['key_as_string']: bucket['doc_count']
            for bucket in results['aggregations']['year-agg']['buckets']
            if bucket['doc_count'] > 0
        }
    }
    
    return render_template(
        'index.html', query=query, results=results['hits']['hits'], from_=from_, total=results['hits']['total']['value'],aggs=aggs)


@app.get('/document/<id>')
def get_document(id):
    document = es.retrieve_document(id)
    title = document['_source']['name']
    paragraphs = document['_source']['content'].split('\n')
    return render_template('document.html',title=title, paragraphs=paragraphs)
    return 'Document not found'


@app.cli.command()
def reindex():
    """Regenerate the elasticsearch index."""
    response = es.reindex()
    print(f'Index with {len(response["items"])} documents created in {response["took"]}ms')
    