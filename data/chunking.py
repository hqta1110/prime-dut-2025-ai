import re
import numpy as np

def sentence_split(text):
    single_sentences_list = re.split('([。；？！.;?!])', text)
    sentences = ["".join(i) for i in  zip(single_sentences_list[0::2], single_sentences_list[1::2])]
    sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences)]
    return sentences

def combine_sentences(sentences, buffer_size=1):
    combined_sentences = [
        " ".join(
            sentences[j]['sentence'] for j in range(max(i - buffer_size, 0), min(i + buffer_size + 1, len(sentences)))
        ) for i in range(len(sentences))
    ]    
    for i, combined_sentence in enumerate(combined_sentences):
        sentences[i]['combined_sentence'] = combined_sentence
    
    return sentences

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_product / (norm_vec1 * norm_vec2)

def calculate_cosine_distance(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i+1]['combined_sentence_embedding']
        similarity = cosine_similarity(embedding_current, embedding_next)
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]['distance_to_next'] = distance

    return distances, sentences 

def chunk_com(distances, bpp_threshold=80):
    breakpoint_percentile_threshold = bpp_threshold
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    indices_above_threshold = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    return indices_above_threshold, breakpoint_distance_threshold


