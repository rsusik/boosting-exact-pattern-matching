from utils import run_tests


datasets = [
    'data/english/english.50MB',
    'data/dna/dna.50MB',
    'data/dblp.xml/dblp.xml.50MB',
    'data/pitches/pitches.50MB',
    'data/proteins/proteins.50MB',
    'data/sources/sources.50MB'
]

algorithms = ['bf', 'kr', 'qs', 'nsn', 'smith', 'rcolussi', 'askip', 'br', 'fs', 'ffs', 'bfs', 'ts', 'ssabs', 'tvsbs', 'fjs', 'hash3', 'hash5', 'hash8', 'aut', 'rf', 'bom', 'bom2', 'ww', 'ildm1', 'ildm2', 'ebom', 'fbom', 'sebom', 'sfbom', 'so', 'sa', 'bndm', 'bndml', 'sbndm', 'sbndm2', 'sbndm-bmh', 'faoso2', 'aoso2', 'aoso4', 'fsbndm', 'bndmq2', 'bndmq4', 'bndmq6', 'sbndmq2', 'sbndmq4', 'sbndmq6', 'sbndmq8', 'ufndmq4', 'ufndmq6', 'ufndmq8']


run_tests(
    r=10,
    algos=algorithms,
    dataset_path='/tmp/datasets',
    copy_datasets=True,
    datasets=datasets
)

