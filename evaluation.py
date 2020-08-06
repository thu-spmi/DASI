import argparse, pickle, csv, os
from eval.eval_model import evaluate_on_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    with open(args.file, 'rb') as f:
        embs = pickle.load(f)
    results = evaluate_on_all(embs)
    print(results)

    results_dict = {'name': args.file.split('/')[-1]}
    for s in results.split():
        results_dict[s.split(':')[0]] = s.split(':')[1]

    write_title = False if os.path.exists('embedding_evaluation.csv') else True
    with open('embedding_evaluation.csv', 'a') as rf:
        writer = csv.DictWriter(rf, fieldnames=list(results_dict.keys()))
        if write_title:
            writer.writeheader()
        writer.writerows([results_dict])