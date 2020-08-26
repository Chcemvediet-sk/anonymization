#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys

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

def us(text):
    if isinstance(text, str) or isinstance(text, bytes):
        return unicode(text,'utf-8')
    elif isinstance(text, unicode):
        return text
    else:
        return unicode(str(text),'utf-8')

def utfstrip(word):
    loword = us(word).lower()
    key = {"a":u"áä","c":u"č","d":u"ď","e":u"éě",u"i":u"í",u"l":u"ľ",u"n":u"ň",
           "o":u"óô","r":u"ř","s":u"š","t":u"ť",u"u":u"ú",u"y":u"ý",u"z":u"ž"}
    asciiset = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
    loword = [i for i in loword]
    for i in range(len(loword)):
        for ii in key:
            if loword[i] in key[ii]:
                loword[i] = ii
        if loword[i] not in asciiset:
            loword[i] = u'.'
    return ''.join(loword)

def wordrecognize(word1,word2,output='bool',threshold=0.3):
    """Using Levenshtein Distance divided by the length of a word we get the similarity of two words
    word1 is the original word, word2 is the one compared to word1.
    """
    word1, word2 = utfstrip(us(word1).lower()), utfstrip(us(word2).lower())
    word2 = ''.join([word2[i] for i in range(len(word2)) if word2[i] not in [',','.','?','!']])

    for end in ['eho','ho','emu','ej','ym','ou']:
        if end in word2[-3:]:
            word1 = word1[:-1]+"."
            word2 = word2[:-(len(end))]+"."
            break

    if len(word1) > len(word2):
        s1, s2 = word2, word1
    else:
        s2, s1 = word2, word1

    insert_costs = np.ones(128, dtype=np.float64)
    delete_costs = np.ones(128, dtype=np.float64)
    substitute_costs = np.ones((128,128), dtype=np.float64)
    delete_costs[ord(",")] = 0
    delete_costs[ord(".")] = 0
    delete_costs[ord("-")] = 0
    insert_costs[ord(",")] = 0
    insert_costs[ord(".")] = 0
    insert_costs[ord("-")] = 0

    similargroups = [["i","j","l","!","1"],["s","f"],["h","b"],["e","c","o"]]
    for group in similargroups:
        for character in group:
            for comcharacter in group:
                substitute_costs[ord(character),ord(comcharacter)] = 0.5
    distance = lev(s1, s2, substitute_costs=substitute_costs, delete_costs=delete_costs, insert_costs=insert_costs)
    if output == 'float':
        return (distance/len(word1))
    if word1 in word2 and len(word1)>2:
        return True
    return True if (distance/len(word1)) <= threshold else False

def streetsplit(string):
    string = string.split()
    street = []
    nearby = []
    housenumber = ''
    for curr_word in string:
        for number in range(10):
            if us(number) in us(curr_word):
                housenumber = curr_word
                string.remove(curr_word)
                break
        if housenumber != '':
            break
        if len(curr_word) > 3:
            street.append(''.join([curr_word[i] for i in range(len(curr_word)) if curr_word[i] != ',']))
        else:
            nearby.append(curr_word)
    return [street,nearby,housenumber]

def substitute_identity(content,name,street,city,zipcode,anonymization_string=u"xxxxxxx"):
    """Replace all instances of personal information in text.
    """
    content = us(content)
    name,street,city,zipcode = us(name),us(street),us(city),us(zipcode)
    text = content
    content = content.split()
    censoring = []
    
    cities = ['bratislava','kosice','presov','zilina','banska bystrica',
               'nitra','trnava','trencin','martin','poprad','prievidza',
              'zvolen','povazska bystrica','michalovce','nove zamky',
              'spisska nova ves','komarno',' humenne','levice','bardejov']
    
    firstname = name.split()[0]
    surname = name.split()[-1]
    streetname = streetsplit(street)[0]
    street_attribute = streetsplit(street)[1]
    street_number = streetsplit(street)[2]
    for index in range(len(content)):
        curr_word = content[index]
        if index==0:
            preword = ''
        else:
            preword = content[index-1]
        if index==(len(content)-1):
            afterword = ''
        else:
            afterword = content[index+1]
        if wordrecognize(surname,curr_word):         
            preword_float = wordrecognize(firstname,preword,'float')
            afterword_float = wordrecognize(firstname,afterword,'float')
            
            if preword_float < afterword_float and preword_float < 0.45:
                censoring.append(preword+" "+curr_word)
            elif afterword_float < preword_float and afterword_float < 0.45:
                censoring.append(curr_word+" "+afterword)
            else:
                censoring.append(curr_word)
        for street_iter in streetname:
            if wordrecognize(street_iter,curr_word):
                for attribute_iter in street_attribute:
                    preword_float = wordrecognize(attribute_iter,preword,'float')
                    afterword_float = wordrecognize(attribute_iter,afterword,'float')
                    if preword_float < afterword_float and preword_float < 0.35:
                        censoring.append(preword+" "+curr_word)
                        street_attribute.remove(attribute_iter)
                    elif afterword_float < preword_float and afterword_float < 0.35:
                        censoring.append(curr_word+" "+afterword)
                        street_attribute.remove(attribute_iter)
                    else:
                        censoring.append(curr_word)
                if len(street_attribute) < 1:
                  censoring.append(curr_word)
        if wordrecognize(street_number,curr_word):
            censoring.append(curr_word)
        if len(city.split()) == 1:
            cityword = curr_word
        if len(city.split()) == 2:
            cityword = curr_word + " " +afterword
        if len(city.split()) > 2:
            cityword = preword + ' ' + curr_word + ' ' + afterword
            city = ' '.join(city.split()[:3])
        if wordrecognize(city, cityword):
            go_on = True
            for city_iter in cities:
                if wordrecognize(city,city_iter):
                    go_on = False
            if go_on:
                censoring.append(cityword)
        if wordrecognize(zipcode,curr_word):
            censoring.append(curr_word)
        elif len(zipcode)==5:
            zipkod = [zipcode[:3],zipcode[3:]]
            if wordrecognize(zipkod[0],curr_word) and wordrecognize(zipkod[1],afterword):
                censoring.append(curr_word+' '+afterword)
    anonymized = text
    for confidential in sorted(censoring, key=len, reverse=True):
        anonymized = anonymized.replace(confidential,us(anonymization_string))
    return anonymized

def anonymize_markup_new(content, parser, name, street, city, zipcode, xpath=u'.//', namespace=None):
    u"""
    Anonymize user in each xpath of markup (xml or html) content, using defined namespace.
    """
    root = etree.fromstring(content, parser) #text odt suboru je ako XML strom
    for t in root.findall(xpath, namespace):
        for tt in list(t): 
            if tt.tail is None:
                continue
            tt.tail = substitute_identity(tt.tail,name,street,city,zipcode,ANONYMIZATION_STRING)
        if t.text is None:
            continue
        t.text = substitute_identity(t.text,name,street,city,zipcode,ANONYMIZATION_STRING)
    return etree.tostring(root)

def anonymize_odt(filename, filenameout, name, street, city, zipcode):
    try:
        parser = etree.XMLParser() #inicializuje parser
        namespace = {u'text': u'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
        filein = open(filename,"r")
        fileout = open(filenameout,"w")
        filecontent = filein.read()

        with closing(StringIO.StringIO(filecontent)) as buffer_in: #vstupny subor sa cely nacita
            with closing(StringIO.StringIO()) as buffer_out: #sem pride vystup, do ktoreho sa da nasledovne:
                with zipfile.ZipFile(buffer_in) as zipfile_in: #rozbalime vstupny subor 
                    with zipfile.ZipFile(buffer_out, u'w') as zipfile_out: #vystupny sybor bude zabaleny
                        for f in zipfile_in.filelist: # pre kazdy subor zo zipka (vstupneho) - ODT je totiz ZIP
                            content = zipfile_in.read(f)
                            if magic.from_buffer(content, mime=True) in content_types.XML_CONTENT_TYPES: #ak je to text ODT suboru, tak ho anonymizujeme
                                zipfile_out.writestr(f, anonymize_markup_new(
                                         content, parser, name, street, city, zipcode, u'.//text:span', namespace))
                            else:
                                zipfile_out.writestr(f, content)
                fileout.write(buffer_out.getvalue())
    except Exception as e:
        trace = unicode(traceback.format_exc(), u'utf-8')
        print(trace)
        print(e)



anonymize_odt(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])
#print unicode(sys.argv[1],"utf-8")
