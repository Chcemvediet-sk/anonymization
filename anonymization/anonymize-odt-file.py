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
    result = []
    for i in range(len(loword)):
        for ii in key:
            if loword[i] in key[ii]:
                loword[i] = ii
        if loword[i] not in asciiset:
            loword[i] = '.'
    return ''.join(loword)

def WordRecognize(slovo1,slovo2,output="bool"):
    """Cez levenshteinovu vzdialenost zistujeme pocet krokov od jednoho slova k druhemu,
    prve slovo1 musi byt vzdy nas original udaj, slovo2 je vyraz s ktorym ho porovnavame
    """
    slovo1, slovo2 = utfstrip(us(slovo1).lower()), utfstrip(us(slovo2).lower())
    slovo2 = ''.join([slovo2[i] for i in range(len(slovo2)) if slovo2[i] not in [',','.','?','!']])
    for end in ['eho','ho','emu','ej','ym','ou']:
        if end in slovo2[-3:]:
            slovo1 = slovo1[:-1]+"."
            slovo2 = slovo2[:-(len(end))]+"."
            break
    if len(slovo1) > len(slovo2):
        s1, s2 = slovo2, slovo1
    else:
        s2, s1 = slovo2, slovo1
    insert_costs = np.ones(128, dtype=np.float64)
    delete_costs = np.ones(128, dtype=np.float64)
    substitute_costs = np.ones((128,128), dtype=np.float64)
    delete_costs[ord(",")] = 0
    delete_costs[ord(".")] = 0
    delete_costs[ord("-")] = 0
    insert_costs[ord(",")] = 0
    insert_costs[ord(".")] = 0
    insert_costs[ord("-")] = 0

    skupiny=[["i","j","l","!","1"],["s","f"],["h","b"],["e","c","o"]]
    for skupina in skupiny:
        for pismeno in skupina:
            for compismeno in skupina:
                substitute_costs[ord(pismeno),ord(compismeno)] = 0.5
    distance = lev(s1, s2, substitute_costs=substitute_costs, delete_costs=delete_costs, insert_costs=insert_costs)

    if output == "float":
        return distance/len(slovo1)
    if slovo1 in slovo2 and len(slovo1)>2:
        return True
    return True if (distance/len(slovo1)) <= 0.3 else False

def StreetSplitter(street):
    street = street.split()
    ulica = []
    prilahle = []
    cislodomu = ''
    for slovo in street:
        for number in range(10):
            if us(number) in us(slovo):
                cislodomu = slovo
                street.remove(slovo)
                break
        if cislodomu != '':
            break
        if len(slovo) > 3:
            ulica.append(''.join([slovo[i] for i in range(len(slovo)) if slovo[i] != ',']))
        else:
            prilahle.append(slovo)
    return [ulica,prilahle,cislodomu]

def Substitute_Identity(content,meno,street,mesto,zipcode,anonymization_string=u"xxxxxxx"):
    """Replace all instances of personal information in text.
    """
    content = us(content)
    meno,street,mesto,zipcode = us(meno),us(street),us(mesto),us(zipcode)
    text = content
    content = content.split()
    Censoring = []
    
    VelkoMesta = ['bratislava','kosice','presov','zilina','banska bystrica',
                  'nitra','trnava','trencin','martin','poprad','prievidza',
                  'zvolen','povazska bystrica','michalovce','nove zamky',
                  'spisska nova ves','komarno',' humenne','levice','bardejov']
    
    krstne = meno.split()[0]
    priezvisko = meno.split()[1]
    ulica = StreetSplitter(street)[0]
    streetprivlastok = StreetSplitter(street)[1]
    streetnumber = StreetSplitter(street)[2]
    for index in range(len(content)):
        slovo = content[index]
        if index==0:
            predword = ''
        else:
            predword = content[index-1]
        if index==(len(content)-1):
            poword = ''
        else:
            poword = content[index+1]
        if WordRecognize(priezvisko,slovo):         
            predslovo = WordRecognize(krstne,predword,'float')
            poslovo = WordRecognize(krstne,poword,'float')
            
            if predslovo < poslovo and predslovo < 0.45:
                Censoring.append(predword+" "+slovo)
            elif poslovo < predslovo and poslovo < 0.45:
                Censoring.append(slovo+" "+poword)
            else:
                Censoring.append(slovo)
        for nazovulice in ulica:
            if WordRecognize(nazovulice,slovo):
                for privlastok in streetprivlastok:
                    predslovo = WordRecognize(privlastok,predword,'float')
                    poslovo = WordRecognize(privlastok,poword,'float')
                    if predslovo < poslovo and predslovo < 0.35:
                        Censoring.append(predword+" "+slovo)
                        streetprivlastok.remove(privlastok)
                    elif poslovo < predslovo and poslovo < 0.35:
                        Censoring.append(slovo+" "+poword)
                        streetprivlastok.remove(privlastok)
                    else:
                        Censoring.append(slovo)
                if len(streetprivlastok) < 1:
                  Censoring.append(slovo)
        if WordRecognize(streetnumber,slovo):
            Censoring.append(slovo)
        if len(mesto.split()) == 1:
            mestoslovo = slovo
        if len(mesto.split()) == 2:
            mestoslovo = slovo + " " +poword
        if len(mesto.split()) > 2:
            mestoslovo = predword + ' ' + slovo + ' ' + poword
            mesto = ' '.join(mesto.split()[:3])
        if WordRecognize(mesto, mestoslovo):
            ideme = True
            for velkomesto in VelkoMesta:
                if WordRecognize(mesto,velkomesto):
                    ideme = False
            if ideme:
                Censoring.append(mestoslovo)
        if WordRecognize(zipcode,slovo):
            Censoring.append(slovo)
        elif len(zipcode)==5:
            zipkod = [zipcode[:3],zipcode[3:]]
            if WordRecognize(zipkod[0],slovo) and WordRecognize(zipkod[1],poword):
                Censoring.append(slovo+' '+poword)
    anonymized = text
    for udaj in sorted(Censoring, key=len, reverse=True):
        anonymized = anonymized.replace(udaj,us(anonymization_string))
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
            tt.tail = Substitute_Identity(tt.tail,name,street,city,zipcode,ANONYMIZATION_STRING)
        if t.text is None:
            continue
        t.text = Substitute_Identity(t.text,name,street,city,zipcode,ANONYMIZATION_STRING)
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
