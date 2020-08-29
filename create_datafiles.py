import sys
import os
import hashlib
import struct, time
import subprocess
import collections
from tensorflow.core.example import example_pb2
import pandas as pd


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

raw_dir= 'dataset/raw'
tokenized_dir= 'dataset/tokenized'
final_file_dir= 'dataset/final'
chunk_final= 'dataset/chunked'

VOCAB_SIZE= 25000
CHUNK_SIZE= 1000


def csv_to_txt(csv_file):
    file= pd.read_csv(csv_file, sep=',', encoding='utf-8')
    para_file= open("dataset/raw/paragraph.txt",'w')
    summ_file= open('dataset/raw/summary.txt','w')
    for i in range(len(file['dialogue-hinglish'])): #change this
        cp= str(file['dialogue-hinglish'][i]).replace('\n',' ')
        cs= str(file['summary-english'][i]).strip()
        para_file.write(cp+"\n")
        summ_file.write(cs+"\n")
    
    para_file.close()
    summ_file.close()


def tokenize_files(raw_dir, tokenized_dir):
    """Maps a whole directory of .txt files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (raw_dir, tokenized_dir))
    stories = os.listdir(raw_dir)
    # make IO list file
    print("Making list of files to tokenize...", stories)
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(raw_dir, s), os.path.join(tokenized_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), raw_dir, tokenized_dir))
    subprocess.call(command)
    # time.sleep(5)
    os.remove("mapping.txt")
    print("Stanford CoreNLP Tokenizer has finished.")

    # Check that the tokenized directory contains the same number of files as the original directory
    num_orig = len(os.listdir(raw_dir))
    num_tokenized = len(os.listdir(tokenized_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_dir, num_tokenized, raw_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (raw_dir, tokenized_dir))


def write_to_bin(start_idx, end_idx, out_file, makevocab=False):
    """Reads the tokenized files corresponding to the urls listed in the url_file and writes them to a out_file."""
    print("Making bin file ")
    tok_para_file= open(os.path.join(tokenized_dir,"paragraph.txt"),'r')
    tok_summ_file= open(os.path.join(tokenized_dir,"summary.txt",), 'r')
    req_para= tok_para_file.readlines()[start_idx: end_idx]
    req_summ= tok_summ_file.readlines()[start_idx: end_idx]
    tok_para_file.close()
    tok_summ_file.close()

    if makevocab:
        vocab_counter = collections.Counter()
    print(len(req_para), len(req_summ))
    with open(out_file, 'wb') as writer:
        for idx in range(len(req_para)):
            if idx % 1000 == 0:
                print("Writing story  ", idx)

            # Look in the tokenized story dirs to find the .story file corresponding to this url

            # Get the strings to write to .bin file
            article, abstract = req_para[idx], req_summ[idx]

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([bytes(article, encoding='utf-8')])
            tf_example.features.feature['abstract'].bytes_list.value.extend([bytes(abstract, encoding='utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if
                              t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print( "Writing vocab file...")
        with open(os.path.join(final_file_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")



def chunk_file(set_name):
    in_file = 'dataset/final/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunk_final, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunk_final):
        os.mkdir(chunk_final)
    # Chunk the data
    for set_name in ['train', 'valid', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunk_final)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise AttributeError("use as python create_datafile /path/to/data.csv")
        sys.exit()
    csv_dir= sys.argv[1]

    if not os.path.exists(tokenized_dir): os.makedirs(tokenized_dir)
    if not os.path.exists(final_file_dir): os.makedirs(final_file_dir)
    if not os.path.exists(chunk_final): os.makedirs(chunk_final)
    if not os.path.exists(raw_dir): os.makedirs(raw_dir)
    #convert csv to text file of paragraph and summary
    csv_to_txt(csv_dir)
    
    # tokenize paragraph and summary
    # tokenize_files(raw_dir=raw_dir, tokenized_dir= tokenized_dir)
    #bineraize files and split in train, valid and test
    #test

    print("binartjufds")
    write_to_bin(start_idx=500, end_idx=1000, out_file= os.path.join(final_file_dir, "test.bin"), makevocab= False)
    #vaalid
    write_to_bin(start_idx=0, end_idx=500, out_file= os.path.join(final_file_dir, "valid.bin"), makevocab= False)
    #train file and save vocab
    write_to_bin(start_idx=1000, end_idx=-1, out_file= os.path.join(final_file_dir, "train.bin"), makevocab= True)
    #chunk train file in to chunks of 1000
    chunk_all()
        








