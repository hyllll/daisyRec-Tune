from app import app
from flask import render_template

score_metric = [
    {'name': 'ndcg'},
    {'name': 'precision'},
    {'name': 'recall'},
    {'name': 'hr'},
    {'name': 'map'},
    {'name': 'mrr'},
]
problem_type = [
    {'name': 'point'},
    {'name': 'pair'},
]
algo_name = [
    {'name': 'vae'},
    {'name': 'cdae'},
    {'name': 'mf'},
    {'name': 'fm'},
    {'name': 'nfm'},
    {'name': 'afm'},
    {'name': 'deepfm'},
    {'name': 'neumf'},
    {'name': 'item2vec'},
    {'name': 'userknn'},
    {'name': 'itemknn'},
    {'name': 'pop'},
    {'name': 'slim'},
    {'name': 'puresvd'}
]
dataset_name = [
    {'name': 'ml-100k'},
    {'name': 'ml-1m'},
    {'name': 'ml-10m'},
    {'name': 'netflix'},
    {'name': 'lastfm'},
    {'name': 'bx'},
    {'name': 'amazon-cloth'},
    {'name': 'amazon-book'},
    {'name': 'amazon-music'},
    {'name': 'epinions'},
    {'name': 'yelp'},
    {'name': 'citeulike'},
]
preprocess_methods = [
    {'name': 'origin'},
    {'name': '5core'},
    {'name': '10core'},
]
test_methods = [
    {'name': 'tloo (time-aware leave one out)'},
    {'name': 'loo (leave one out)'},
    {'name': 'tfo (time-aware fold-out)'},
    {'name': 'fo (fold-out)'},
    {'name': 'utfo (time-aware fold-out withuser-level)'},
    {'name': 'ufo (fold-out with user-level)'},
]
val_methods = [
    {'name': 'tloo'},
    {'name': 'loo'},
    {'name': 'tfo'},
    {'name': 'tfo'},
    {'name': 'utfo'},
    {'name': 'ufo'},
    {'name': 'cv'},
]
sample_methods = [
    {'name': 'uniform (uniformly sampling)'},
    {'name': 'item-ascd (sampling popular items with low rank)'},
    {'name': 'item-desc (sampling popular item with high rank)'},
]
loss_types = [
    {'name': 'CL (Cross-entropy Loss)'},
    {'name': 'SL (Square Loss)'},
    {'name': 'BPR (BPR Loss)'},
    {'name': 'HL (Hinge Loss)'},
    {'name': 'TL (Top-1 Loss)'},
]

@app.route('/')
@app.route('/main')
def main():
    user = {'username':'Daisy'}
    

    return render_template(
        'main.html', 
        title='yudi', 
        user=user, 
        score_metric=score_metric,
        problem_type=problem_type,
        algo_name=algo_name,
        dataset_name=dataset_name,
        preprocess_methods=preprocess_methods,
        test_methods=test_methods,
        val_methods=val_methods,
        sample_methods=sample_methods,
        loss_types=loss_types
    )

@app.route('/hpo_tuner')
def hpo_tuner():
    user = {'username':'Daisy'}
    return render_template(
        'hpo_tuner.html', 
        title='yudi', 
        user=user, 
        score_metric=score_metric,
        problem_type=problem_type,
        algo_name=algo_name,
        dataset_name=dataset_name,
        preprocess_methods=preprocess_methods,
        test_methods=test_methods,
        val_methods=val_methods,
        sample_methods=sample_methods,
        loss_types=loss_types
    )