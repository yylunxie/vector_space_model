import xml.etree.ElementTree as ET

def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # skip encoding line
    return [line.strip() for line in lines]

def load_file_list(file_list_path):
    with open(file_list_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_inverted_index(inverted_path):
    inverted_index = {}
    doc_lens = {}

    with open(inverted_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        vocab_id1, vocab_id2, N = int(parts[0]), int(parts[1]), int(parts[2])
        term_key = (vocab_id1, vocab_id2)
        inverted_index[term_key] = {}
        i += 1
        for _ in range(N):
            file_id, count = map(int, lines[i].strip().split())
            inverted_index[term_key][file_id] = count
            doc_lens[file_id] = doc_lens.get(file_id, 0) + count
            i += 1

    return inverted_index, doc_lens

def parse_query_xml(query_path):
    tree = ET.parse(query_path)
    root = tree.getroot()
    queries = {}
    for topic in root.findall('topic'):
        num = topic.find('number').text.strip()[-3:]  # e.g. '001'
        title = topic.find('title').text.strip()
        question = topic.find('question').text.strip()
        narrative = topic.find('narrative').text.strip()
        concepts = topic.find('concepts').text.strip()
        # You can modify this part to decide what fields to use
        query_text = title + ' ' + question + ' ' + narrative + ' ' + concepts
        queries[num] = query_text
    return queries
