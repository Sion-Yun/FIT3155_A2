__author__ = "Yun Sion"

# Github = https://github.com/Sion-Yun/FIT3155_A2

import heapq
from collections import Counter
import sys

# TODO - comment, docstring

####################
# Util
####################

def convert_binary(n):
    binary_str = ""
    while n >= 1:
        if n % 2 == 0:
            binary_str += "0"
        else:
            binary_str += "1"
        n //= 2
    return binary_str[::-1]

def elias_encoding(n):
    # function implementation to convert a given value into elias encoding
    encoded_binary = ""
    # Plus 1 for the non-negativeness of Elias codeword (adjusting index to start from 1)
    binary = convert_binary(n+1)
    encoded_binary += binary
    k = len(binary) - 1

    while k >= 1:
        front_binary = convert_binary(k)
        front_binary = "0" + front_binary[1:len(front_binary)]
        encoded_binary = front_binary + encoded_binary
        k = len(front_binary) - 1
    return encoded_binary


class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    # Comparison operators for heapq
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    # Build the Huffman tree
    freq = Counter(text)
    heap = [HuffmanNode(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)

    return heap[0]  # Return the root node

def generate_huffman_codes(node, prefix="", huffman_codes={}):
    # Recursively generate the Huffman codes, starting from the root
    if node.char is not None:
        huffman_codes[node.char] = prefix
    if node.left is not None:
        generate_huffman_codes(node.left, prefix + "0", huffman_codes)
    if node.right is not None:
        generate_huffman_codes(node.right, prefix + "1", huffman_codes)
    return huffman_codes

def lz77(string):
    #  encoding chars (= string) into lz77 format
    encoded_arr = []
    k = 0

    while k < len(string):
        pointer = k - 1
        match_count = []

        while pointer >= 0:
            match_counter = 0
            for i in range(len(string) - k):
                if string[pointer] != string[k]:
                    break
                if k + i <= len(string) - 1 and string[pointer + i] == string[k + i]:
                    match_counter += 1
                else:
                    break
            match_count.append([match_counter, k - pointer])
            if match_counter == len(string) - k:  # if the number of matches is same as the buffer size, break
                break
            pointer -= 1

        # Find the maximum match and its offset
        max_match, offset = max(match_count, key=lambda x: x[0]) if match_count else (0, 0)

        # Include the next character after the match, regardless of the match length
        next_char = string[k + max_match] if k + max_match < len(string) else ''

        # Append the tuple ⟨offset, length, next_char⟩
        encoded_arr.append([offset, max_match, next_char])

        # Move past the match
        k += max_match + 1  # Always move past the match + next_char

    # print("encoded arr from lz77: ", encoded_arr)
    return encoded_arr  # return an array in lz77 format

###############################
# Encoding
###############################

class Encode:
    def __init__(self, filename, txt):
        self.file_size = len(txt)

        # LZ77 compression
        self.lzss_tuples = lz77(txt)

        # Huffman coding
        self.huffman_codes = generate_huffman_codes(build_huffman_tree(txt))

        # Combine header and LZ77 encoded data
        encoded_output = self.encode_header(filename) + self.encode_lz77_data()
        # print(encoded_output)

        # Write to binary file
        output_filename = filename.split('.')[0] + ".bin"
        # output_filename = filename.split('.')[0] + ".txt"
        self.output(encoded_output, output_filename)

    def encode_header(self, filename):
        encoded_header = ""

        # 1. Encode file size using Elias encoding
        # print(elias_encoding(self.file_size))
        encoded_header += elias_encoding(self.file_size)

        # 2. Encode filename length using Elias encoding
        filename_length = len(filename)
        # print(elias_encoding(filename_length))
        encoded_header += elias_encoding(filename_length)

        # 3. Encode the filename using 8-bit ASCII
        for char in filename:
            # print(convert_binary(ord(char)).zfill(8))
            encoded_header += convert_binary(ord(char)).zfill(8)

        # 4. Encode the number of distinct characters
        distinct_chars = len(self.huffman_codes)
        # print(elias_encoding(distinct_chars))
        encoded_header += elias_encoding(distinct_chars)

        # 5. For each character, encode its ASCII, codeword length, and the Huffman code
        for char, code in self.huffman_codes.items():
            encoded_header += convert_binary(ord(char)).zfill(8)  # 8-bit ASCII
            encoded_header += elias_encoding(len(code))  # Codeword length
            encoded_header += code  # Huffman code itself
            """
            this code gives distinct chars in non-binary order,  (e.g. b = 00, c = 01, a = 1)
            this differs from the A2 specs example but this could be encoded in any order, according to the specs
            """
            # print(char, convert_binary(ord(char)).zfill(8), elias_encoding(len(code)), code)
        return encoded_header

    def encode_lz77_data(self):
        encoded_data = ""

        for tuple in self.lzss_tuples:
            # print(tuple)
            if len(tuple) == 3:  # Tuple ⟨offset, length, next_char⟩
                offset, length, next_char = tuple
                encoded_data += elias_encoding(offset)  # offset using Elias
                encoded_data += elias_encoding(length)  # length using Elias
                encoded_data += self.huffman_codes[next_char]  # next_char using Huffman
                # print(elias_encoding(offset), elias_encoding(length), self.huffman_codes[next_char])
            else:  # Tuple of form ⟨literal⟩
                literal = tuple[0]
                encoded_data += self.huffman_codes[literal]
                # print(self.huffman_codes[literal])
        return encoded_data

    def output(self, encoded_bits, output_filename):
        # print(encoded_bits)
        byte_arr = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte = encoded_bits[i:i + 8]
            byte_arr.append(int(byte, 2))

        with open(output_filename, "wb") as f:
            f.write(byte_arr)


if __name__ == "__main__":
    # TODO - test
    # python a2q2.py <asc filename>
    file_name = sys.argv[1]
    with open(file_name, 'r') as f:
        text = f.read()

    Encode(file_name, text)

    # file_name = "x.asc"
    # text = "aacaacabcaba"