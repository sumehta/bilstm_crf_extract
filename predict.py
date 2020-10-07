import os, sys
import pandas as pd
# import ipdb; ipdb.set_trace()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from bi_lstm_crf.app.preprocessing import *
from bi_lstm_crf.app.utils import *



class WordsTagger:
    def __init__(self, model_dir, device=None):
        args_ = load_json_file(arguments_filepath(model_dir))
        args = argparse.Namespace(**args_)
        args.model_dir = model_dir
        self.args = args

        self.preprocessor = Preprocessor(config_dir=model_dir, verbose=False)
        self.model = build_model(self.args, self.preprocessor, load=True, verbose=False)
        self.device = running_device(device)
        self.model.to(self.device)

        self.model.eval()

    def __call__(self, sentences, begin_tags="BS"):
        """predict texts

        :param sentences: a text or a list of text
        :param begin_tags: begin tags for the beginning of a span
        :return:
        """
        if not isinstance(sentences, (list, tuple)):
            raise ValueError("sentences must be a list of sentence")

        try:
            max_seq_len = max([len(sent) for sent in sentences])
            sent_tensor = np.array([np.array(self.preprocessor.sent_to_vector(s, max_seq_len=max_seq_len)) for s in sentences])
            sent_tensor = torch.from_numpy(sent_tensor).to(self.device)
            with torch.no_grad():
                _, tags = self.model(sent_tensor)
            tags = self.preprocessor.decode_tags(tags)
        except RuntimeError as e:
            print("*** runtime error: {}".format(e))
            raise e
        return tags, self.tokens_from_tags(sentences, tags, begin_tags=begin_tags)

    @staticmethod
    def tokens_from_tags(sentences, tags_list, begin_tags):
        """extract entities from tags

        :param sentences: a list of sentence
        :param tags_list: a list of tags
        :param begin_tags:
        :return:
        """
        if not tags_list:
            return []

        def _tokens(sentence, ts):
            actors = []
            targets = []
            for i in range(len(ts)):
                if 'B-ACT' in ts[i]:
                    j = i+1
                    while j < len(ts) and 'ACT' in ts[j]:
                        j+=1
                    actors.append((' '.join(sentence[i:j]), ' '.join(ts[i:j])))

            for i in range(len(ts)):
                if 'B-TARG' in ts[i] or 'I-TARG' in ts[i]:
                    j = i+1
                    while j < len(ts) and 'TARG' in ts[j]:
                        j+=1
                    targets.append((' '.join(sentence[i:j]), ' '.join(ts[i:j])))

            return (actors, targets)

        tokens_list = [_tokens(sentence, ts) for sentence, ts in zip(sentences, tags_list)]
        return tokens_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, help="the sentence to be predicted")
    parser.add_argument('--model_dir', type=str, required=True, help="the model directory for model files")
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')

    args = parser.parse_args()

    sentences = []
    all_results = []


    # for icews
    df = pd.read_json('/home/sudo777/EE/qa_dataset/qa_dataset_15_v5_test.json', lines=True)
    for idx, row in df.iterrows():

        # for icews
        sentences.append(row['sentence'].split())

    bsz = 50
    num_batches = len(sentences)//bsz
    for b_i in range(num_batches):
        if b_i == num_batches-1:
            batch = sentences[b_i*bsz:]
        else:
            batch = sentences[b_i*bsz:(b_i+1)*bsz]
        results = WordsTagger(args.model_dir, args.device)(batch)
        print(len(results[1]))
        for res in results[1]:
            all_results.append(res)

    print('Total records processed: {}'.format(len(all_results)))
    # for icews
    with open('/home/sudo777/EE/bi_lstm_crf/results/icews_act_targ_test_v5.json', 'w+') as f:
        json.dump(all_results, f)



    for objs in results:
        print(json.dumps(objs[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
