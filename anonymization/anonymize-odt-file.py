#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from typing import List, Union, Tuple

import zipfile
import StringIO
import traceback
from contextlib import closing
import content_types
import magic
from lxml import etree
import numpy as np
from weighted_levenshtein import lev
ANONYMIZATION_STRING = u"XXXXX"


def us(text: Union[str, int, bytes]) -> str:
    if isinstance(text, str) or isinstance(text, bytes):
        return unicode(text, 'utf-8')

    if isinstance(text, unicode):
        return text

    return unicode(str(text), 'utf-8')


def utfstrip(word: str) -> str:
    low_caps_word = us(word).lower()
    key = {"a": u"áä", "c": u"č", "d": u"ď", "e": u"éě", u"i": u"í", u"l": u"ľ", u"n": u"ň",
           "o": u"óô", "r": u"ř", "s": u"š", "t": u"ť", u"u": u"ú", u"y": u"ý", u"z": u"ž"}
    asciiset = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
    chars_in_word = [i for i in low_caps_word]
    for i in range(len(chars_in_word)):
        for ii in key:
            if chars_in_word[i] in key[ii]:
                chars_in_word[i] = ii
        if chars_in_word[i] not in asciiset:
            chars_in_word[i] = u'.'
    return ''.join(chars_in_word)


def wordrecognize(word1: str, word2: str, output='bool', threshold=0.3):
    """Using Levenshtein Distance divided by the length of a word we get the
    similarity of two words word1 is the original word, word2 is the one
    compared to word1.
    """

    # if we try recognizing an empty string - the function returns False
    if len(word1) == 0 and output == 'bool':
        return False
    if len(word1) == 0 and output == 'float':
        return 1

    assert output == 'bool' or output == 'float'

    word1, word2 = utfstrip(us(word1).lower()), utfstrip(us(word2).lower())
    word2 = word2.strip(",.?!")

    for end in ['eho', 'ho', 'emu', 'ej', 'ym', 'ou']:
        if end in word2[-3:]:
            word1 = word1[:-1]+"."
            word2 = word2[:-(len(end))]+"."
            break

    if len(word1) > len(word2):
        s1, s2 = word2, word1
    else:
        s1, s2 = word1, word2

    insert_costs = np.ones(128, dtype=np.float64)
    delete_costs = np.ones(128, dtype=np.float64)
    substitute_costs = np.ones((128, 128), dtype=np.float64)
    delete_costs[ord(",")] = 0
    delete_costs[ord(".")] = 0
    delete_costs[ord("-")] = 0
    insert_costs[ord(",")] = 0
    insert_costs[ord(".")] = 0
    insert_costs[ord("-")] = 0

    similargroups = [["i", "j", "l", "!", "1"],
                     ["s", "f"],
                     ["h", "b"],
                     ["e", "c", "o"]]

    for group in similargroups:
        for character in group:
            for comcharacter in group:
                substitute_costs[ord(character), ord(comcharacter)] = 0.5

    distance = lev(s1,
                   s2,
                   substitute_costs=substitute_costs,
                   delete_costs=delete_costs,
                   insert_costs=insert_costs)

    # Sometimes we want to output the calculated distance instead of bool
    if output == 'float':
        return distance/len(word1)

    if word1 in word2 and len(word1) > 2:
        return True
    return (distance/len(word1)) <= threshold


def streetsplit(string) -> Tuple[List[str], List[str], str]:
    string = string.split()

    street = []
    nearby = []
    housenumber = ''

    for curr_word in string:
        curr_word = us(curr_word)
        curr_word = curr_word.strip(',.')

        for num in range(10):
            number = us(num)

            if number in curr_word:
                housenumber = curr_word
                string.remove(curr_word)
                break

        if housenumber != '':
            break

        if len(curr_word) > 3:
            street.append(curr_word)
        else:
            nearby.append(curr_word)
    return (street, nearby, housenumber)


def substitute_identity(content: str,
                        name: str,
                        street: str,
                        city: str,
                        zipcode: str,
                        anonymization_string=u"xxxxxxx") -> str:
    """Replace all instances of personal information in text.
    """
    content = us(content)
    anonymization_string = us(anonymization_string)

    name, street, city, zipcode = us(name), us(street), us(city), us(zipcode)
    text = content
    words_in_content = content.split()
    lex_index = []
    secon_ind = 0

    for i in range(len(words_in_content)):
        first_ind = text[secon_ind:].index(words_in_content[i])+secon_ind
        secon_ind = first_ind + len(words_in_content[i])

        lex_index.append((first_ind, secon_ind))

    censoring = set()

    cities = ['bratislava', 'kosice', 'presov', 'zilina', 'banska bystrica',
              'nitra', 'trnava', 'trencin', 'martin', 'poprad', 'prievidza',
              'zvolen', 'povazska bystrica', 'michalovce', 'nove zamky',
              'spisska nova ves', 'komarno', ' humenne', 'levice', 'bardejov']

    firstname = name.split()[0]
    surname = name.split()[-1]

    streetsplit_result = streetsplit(street)

    streetname = streetsplit_result[0]
    street_attribute = streetsplit_result[1]
    street_number = streetsplit_result[2]

    for index in range(len(words_in_content)):
        curr_word = words_in_content[index]
        curr_index = lex_index[index]

        if index == 0:
            preword = ''
            prev_index = ()
        else:
            preword = words_in_content[index-1]
            prev_index = lex_index[index-1]

        if index == len(words_in_content)-1:
            afterword = ''
            next_index = ()
        else:
            afterword = words_in_content[index+1]
            next_index = lex_index[index+1]

        if wordrecognize(surname, curr_word):
            preword_float = wordrecognize(firstname, preword, 'float')
            afterword_float = wordrecognize(firstname, afterword, 'float')

            if preword_float < afterword_float and preword_float < 0.45:
                censoring.add(prev_index+curr_index)
            elif afterword_float < preword_float and afterword_float < 0.45:
                censoring.add(curr_index+next_index)
            else:
                censoring.add(curr_index)

        for street_iter in streetname:
            # Early return
            if not wordrecognize(street_iter, curr_word):
                continue

            for attribute_iter in street_attribute:
                preword_float = wordrecognize(attribute_iter, preword, 'float')
                afterword_float = wordrecognize(
                    attribute_iter, afterword, 'float')

                if preword_float < afterword_float and preword_float < 0.35:
                    censoring.add(prev_index+curr_index)
                    street_attribute.remove(attribute_iter)
                elif afterword_float < preword_float and afterword_float < 0.35:
                    censoring.add(curr_index+next_index)
                    street_attribute.remove(attribute_iter)
                else:
                    censoring.add(curr_index)

            if len(street_attribute) == 0:
                censoring.add(curr_index)

        if wordrecognize(street_number, curr_word):
            censoring.add(curr_index)

        if len(city.split()) == 1:
            cityword = curr_word
            cityword_index = curr_index
        elif len(city.split()) == 2:
            cityword = curr_word + " " + afterword
            cityword_index = curr_index+next_index
        elif len(city.split()) > 2:
            cityword = preword + ' ' + curr_word + ' ' + afterword
            cityword_index = prev_index+curr_index+next_index
            city = ' '.join(city.split()[:3])

        if wordrecognize(city, cityword):
            go_on = True
            for city_iter in cities:
                if wordrecognize(city, city_iter):
                    go_on = False
            if go_on:
                censoring.add(cityword_index)

        if wordrecognize(zipcode, curr_word):
            censoring.add(curr_index)
        elif len(zipcode) == 5:
            zipkod = [zipcode[:3], zipcode[3:]]
            if wordrecognize(zipkod[0], curr_word) and wordrecognize(zipkod[1], afterword):
                censoring.add(curr_index+next_index)

    anonymized = text
    shift_by = 0
    sorted_censoring = sorted(censoring, key=(lambda x: min(x)))

    for conf_tuple in sorted_censoring:
        if len(conf_tuple) == 0:
            continue

        first_ind = min(conf_tuple)+shift_by
        secondind = max(conf_tuple)+shift_by

        conf_tuple = (first_ind, secondind)
        anonymized = anonymized[:first_ind] + \
            anonymization_string+anonymized[secondind:]
        shift_by += len(anonymization_string)-(secondind-first_ind)

    return anonymized


def anonymize_markup_new(content, parser, name, street, city, zipcode, xpath=u'.//', namespace=None):
    """
    Anonymize user in each xpath of markup (xml or html) content, using defined namespace.
    """
    root = etree.fromstring(content, parser)
    for t in root.findall(xpath, namespace):
        for tt in list(t):
            if tt.tail is None:
                continue
            tt.tail = substitute_identity(
                tt.tail, name, street, city, zipcode, ANONYMIZATION_STRING)
        if t.text is None:
            continue
        t.text = substitute_identity(
            t.text, name, street, city, zipcode, ANONYMIZATION_STRING)
    return etree.tostring(root)


def anonymize_odt(filename, filenameout, name, street, city, zipcode):
    try:
        parser = etree.XMLParser()  # inicializuje parser
        namespace = {
            u'text': u'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
        filein = open(filename, "r")
        fileout = open(filenameout, "w")
        filecontent = filein.read()

        with closing(StringIO.StringIO(filecontent)) as buffer_in:
            with closing(StringIO.StringIO()) as buffer_out:
                with zipfile.ZipFile(buffer_in) as zipfile_in:
                    with zipfile.ZipFile(buffer_out, u'w') as zipfile_out:
                        for f in zipfile_in.filelist:
                            content = zipfile_in.read(f)
                            if magic.from_buffer(content, mime=True) in content_types.XML_CONTENT_TYPES:
                                zipfile_out.writestr(f, anonymize_markup_new(
                                    content, parser, name, street, city, zipcode, u'.//text:span', namespace))
                            else:
                                zipfile_out.writestr(f, content)
                fileout.write(buffer_out.getvalue())
    except Exception as e:
        trace = unicode(traceback.format_exc(), u'utf-8')
        print(trace)
        print(e)


anonymize_odt(sys.argv[1], sys.argv[2], sys.argv[3],
              sys.argv[4], sys.argv[5], sys.argv[6])
