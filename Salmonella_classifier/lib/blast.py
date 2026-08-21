#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified sequence I/O, container, and BLAST compatibility module.

This module combines the former:
    - container.py
    - seq_io.py
    - blast.py

The public APIs are intentionally retained for backward compatibility,
including BLAST, IO, Parser, Fasta, SeqRecord, GBF, GBK, Feature, GFF,
Container, Collection, and the legacy lowercase ``container`` class.

The implementation removes cross-imports between the old modules and uses
one shared namespace.
"""

from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from functools import reduce
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np


# =============================================================================
# Generic containers
# =============================================================================
########################################################################
class Container:
    def __init__(self,title=""):
        self.title = title
        self.container = np.array([], dtype=object)
        self.titles = {}
        self.size = 0
    
    def __add__(self,other):
        if not isinstance(other,Collection):
            raise TypeError("unsupported types of operands!")
        return self.copy().extend(other.get())
        
    def __sub__(self,other):
        if not isinstance(other,Collection):
            raise TypeError("unsupported types of operands!")
        oCopy = self.copy()
        for key in other.get_titles():
            if key in list(self.titles.keys()):
                oCopy.__delitem__(key)
        return oCopy
        
    def __len__(self):
        return self.size
        
    def __contains__(self,key):
        if isinstance(key,str):
            if not self.has(key):
                return False 
        elif isinstance(key,int):
            if key > self.size:
                return False
        else:
            raise ValueError("Container must be refered either by text title or integer index!")
        return True
    
    def __iter__(self):
        if not len(self.container):
            return iter([])
        #return iter(list(map(lambda obj: obj.Obj, self.container)))
        return iter([obj.Obj for obj in self.container])
    
    def __getitem__(self,index):
        if isinstance(index, slice):
            return self.get(index.start,index.stop)
        return self.get(index)
    
    def __setitem__(self,key,Obj):
        index,title = self._parse_key(key)
        if title=="~attribute":
            setattr(self, key, Obj)
            return
        link = None
        if index:
            link = self.container[index-1].Obj.link
        Obj = ContainerElement(Obj,index,link)
        self.titles[title] = Obj
        self.container[index] = Obj
    
    def __delitem__(self,key):
        index,title = self._parse_key(key)
        self.container = np.delete(self.container,index)
        del self.titles[title]
        self.size -= 1
        
    def _parse_key(self,key):
        if not key:
            key = 0
        if isinstance(key,str):
            if not self.has(key):
                if hasattr(self,key):
                    return key,"~attribute"
                raise ValueError(f"There is no object with title {key}!") 
            index = self.titles[key].index
            title = key
        elif isinstance(key,int):
            index = key
            title = self.container[key].title
        else:
            raise ValueError("Container must be refered either by text title or integer index!")
        return index,title
    
    def _get_title(self,obj):
        if hasattr(obj, 'title'):
            return obj.title
        return ""

    def _check_object_title(self,Obj):
        if not hasattr(Obj,'title') or not Obj.title:
            Obj.title = str(self.size)
        if self.has(Obj.title):
            raise KeyError(f"Title {Obj.title} is already occupied!")
        return True
            
    def clear(self):
        self.container = np.array([], dtype=object)
        self.titles = {}
        self.size = 0

    def has(self,title):
        return title in list(self.titles.keys())
    
    def index(self,title):
        if not self.has(title):
            return -1
        return self.titles[title].Obj.index
    
    def append(self,Obj):
        self._check_object_title(Obj)
        link = None
        if self.size:
            link = self.container[self.size-1].link
        Obj = ContainerElement(Obj,self.size,link)
        self.titles[Obj.title] = Obj
        self.container = np.append(self.container, Obj)
        self.size += 1
        
    def insert(self,key,Obj):
        index,title = self._parse_key(key)
        if abs(index) >= self.size:
            self.container.append(Obj)
            return
        self._check_object_title(Obj)
        Obj = ContainerElement(Obj,index,None)
        if index == 0:
            self.container[0].link = Obj.link_function
        else:
            Obj.link = self.container[index-1].link_function
            self.container[index].link = Obj.link_function
        self.container = np.insert(self.container, index, Obj)
        self.titles[Obj.title] = Obj
        self.size += 1            
        
    def extend(self,ls):
        for Obj in ls:
            if hasattr(Obj,'title') and self.has(Obj.title):
                continue
            self.append(Obj)
        
    def get_titles(self):
        return list(self.titles.keys())
    
    def get(self,start_key=None,stop_key=None):
        if start_key==None and stop_key==None:
            #return list(map(lambda obj: obj.Obj.copy(), self.container))
            return [obj.Obj.copy() for obj in self.container]
        index_start,title_start = self._parse_key(start_key)
        if title_start=="~attribute":
            return eval(compile(f"self.{index_start}", "<string>", "eval"))
        if stop_key:
            index_stop,title_stop = self._parse_key(stop_key)
            indices = [index_start,index_stop]
            indices.sort()
            #return list(map(lambda i: self.container[i].Obj, range(indices[0],indices[1]+1,1)))
            return [self.container[i].Obj for i in range(indices[0],indices[1]+1,1)]
        else:
            return self.container[index_start].Obj
            
    def dict(self):
        return self.titles
        
    def push(self,obj_ls):
        if self.size and obj_ls and type(self[0]) != type(obj_ls[0]):
            raise TypeError("Object in the collection and the list are of different types!")
        self.clear()
        for Obj in obj_ls:
            self.append(Obj)
            
    def sort(self,key="",reverse=False):
        obj_list = self.get()
        self.clear()
        # if key==None, sorting is escaped
        if key != None:
            if key=="":
                obj_list.sort(key=lambda Obj: Obj.title)
            else:
                try:
                    eval(compile(f"obj_list.sort(key={key})", "<string>", "eval"))
                except:
                    raise TypeError(f"command {key} cannot be used for sorting these objects!")
        if reverse:
            obj_list.reverse()
        for Obj in obj_list:
            self.append(Obj)
            
    def sorted(self,key="",reverse=False):
        return self.copy().sort(key,reverse)
        
    def reverse(self):
        return self.sorted(key=None,reverse=True)
            
    def copy(self):
        oNewContainer = Collection(self.title)
        for i in range(self.size):
            try:
                oNewContainer.append(self.container[i].Obj.copy())
            except:
                oNewContainer.append(copy.deepcopy(self.container[i].Obj))
        return oNewContainer

########################################################################
class Collection:
    def __init__(self, title=""):
        self.title = title
        self.container = []
        self.para = {}
        
    def _get_key(self,key):
        if type(key)==type(0):
            if len(self) <= key:
                return
            return key
        elif type(key)==type(""):
            return self.index(key)
    
    def _get_title(self,obj):
        if hasattr(obj, 'title'):
            return obj.title
        return ""

    def __len__(self):
        return len(self.container)
        
    def __contains__(self,key):
        if self._get_key(key) != None:
            return True
        return False
    
    def __iter__(self):
        if not self.container:
            return iter([])
        records = []
        for record in self.container:
            records.append(record)
        return iter(records)
    
    def __getitem__(self,key):
        key = self._get_key(key)
        if key != None:
            return self.container[key]
    
    def __setitem__(self,key,value):
        key = self._get_key(key)
        if key != None:
            self.container[key] = value
    
    def __delitem__(self,key):
        key = self._get_key(key)
        if key != None:
            del self.container[key]
            
    def __repr__(self):
        #return "\n".join(list(map(lambda i: "%d\t%s" % (i+1,str(self.container[i])), range(len(self.container)))))
        return "\n".join([f"{i + 1}\t{self.container[i]}" for i in range(len(self.container))])
            
    def __str__(self):
        #return ";".join(list(map(lambda Obj: str(Obj), self.container)))
        return ";".join([str(Obj) for Obj in self.container])
            
    def has(self,title):
        return title in self.get_titles()
    
    def index(self,title):
        if type(title) == type(0):
            index = title
            if abs(index) >= len(self):
                return
            return index
        titles = self.get_titles()
        if title in titles:
            return titles.index(title)
        return
    
    def append(self,obj):
        self.container.append(obj)
        
    def extend(self,ls):
        self.container.extend(ls)
        
    def get_titles(self):
        #return list(map(lambda obj: obj.title, self.container))
        return [obj.title for obj in self.container]
    
    def get(self,titles=[]):
        if titles:
            #return list(filter(lambda Obj: Obj.title in titles, self.container))
            return [Obj for Obj in self.container if Obj.title in titles]
        else:
            return self.container
            
    def copy(self):
        oCollection = Collection(self.title)
        oCollection.para.update(self.para)
        for record in self.container:
            oCollection.append(record.copy())
        return oCollection
            
########################################################################
class ContainerElement:
    def __init__(self,Obj,index,link=None):
        if not hasattr(Obj,'title'):
            Obj.title = str(index)
        self.title = Obj.title
        self.index = index
        self.link = link
        self.Obj = Obj
        
    def link_function(self,command,args=[]):
        if command == "increment":
            self.index += 1
        elif command == "decrement":
            self.index -= 1
        else:
            pass
            

# =============================================================================
# Sequence I/O and sequence-related classes
# =============================================================================
def msg(msg=""):
    print("\n" + str(msg) + "\n")

##############################################################################################################
class IO:
    def __init__(self,path=""):
        self.oParser = Parser(path)
        
    def read(self,path,
                data_format="text",         # text | fasta | genbank | gff | binary
                inlist=False,               # used with 'text', split lines by \n
                separator="",               # used with 'text', split lines by \t
                strip_symbol="",            # used with 'text', strip given symbols
                reverse_complement=False,   # used with FASTA and GenBank to return reverse_complement sequences
                dbkey='$db$',               # used with binary, main key
                splkey='$suppl$',           # used with binary, supplementary key
                    ):
            if data_format.upper()=="TEXT":
                return self.open_text_file(path,inlist,separator,strip_symbol)
            if data_format.upper()=="BINARY":
                return self.open_binary_file(path,dbkey,splkey)
            if data_format.upper() in ("FASTA","FA"<"FST"):
                return self.read_seq(path,seq_format="fasta",reverse_complement=reverse_complement)
            if data_format.upper() in ("GENBANK","GBK","GB"):
                return self.read_seq(path,seq_format="genbank",reverse_complement=reverse_complement)
            if data_format.upper()=="GFF":
                return self.readGFF(path)
        
    def parse(self,path,
                data_format="text",         # text | fasta | genbank | gff | binary
                inlist=False,               # used with 'text', split lines by \n
                separator="",               # used with 'text', split lines by \t
                strip_symbol="",            # used with 'text', strip given symbols
                concatenate=False,          # used with FASTA and GenBank to return concatenated sequences
                reverse_complement=False,   # used with FASTA and GenBank to return reverse_complement sequences
                reverse_contigs=False,      # used with FASTA and GenBank to reverse order of contigs
                dbkey='$db$',               # used with binary, main key
                splkey='$suppl$',           # used with binary, supplementary key
                    ):
            if data_format.upper()=="TEXT":
                return self.open_text_file(self,path,inlist,separator,strip_symbol)
            if data_format.upper()=="BINARY":
                return self.open_binary_file(path,dbkey,splkey)
            if data_format.upper() in ("FASTA","FA"<"FST"):
                return self.parse_seq(path,seq_format="fasta",reverse_complement=reverse_complement,concatenate=concatenate,reverse_contigs=reverse_contigs)
            if data_format.upper() in ("GENBANK","GBK","GB"):
                return self.parse_seq(path,seq_format="genbank",reverse_complement=reverse_complement,concatenate=concatenate,reverse_contigs=reverse_contigs)
            if data_format.upper()=="GFF":
                return self.readGFF(path)
    
    # Append records to file
    def write(self, record, data_format, mode='a'):
        self.save(data=record, data_format=data_format, mode=mode)
        
    #### Collection of save/open functions
    # fasta - FASTA formated collection of sequences    
    def save(self,data,path,
                data_format="text",     # text | binary
                dbkey='$db$',           # used with binary, main key
                splkey='$suppl$',       # used with binary, supplementary key
                suppl_data=None,        # used with binary, supplementary data
                mode="w"                # saving mode, 'w' or 'a'
                    ):
        if data_format=="binary":
            return self.save_binary_file(data,path,dbkey,suppl_data,splkey)
        with open(path, mode) as ofp:
            ofp.write(data)
            ofp.flush()
        return path
    
    def open_text_file(self,path,flg_inlist=False,sep="",strip_symbol=""):
        if not os.path.exists(path):
            return ""
        f = open(path)
        strText = f.read()
        f.close()
        strText = strText.replace("\"","")
        if flg_inlist:
            strText = strText.split("\n")
            if strip_symbol:
                #strText = list(map(lambda line: line.strip(strip_symbol), strText))
                strText = [line.strip(strip_symbol) for line in strText]
            if sep:
                #strText = list(map(lambda item: item.split(sep), strText))
                strText = [line.split(sep) for line in strText]
        return strText
    
    # Parse GFF file
    def readGFF(self, path, mode="object"): # mode = "object" | "dictionary"
        def get_entry(ls):
            # Convert GFF fields into dictionary fields
            entry = dict(zip(["genome","method","modtype","start","end","score","strand","para","data"],ls))
            data = [s.split("=") for s in entry["data"].split(";")]
            entry["data"] = dict(zip(map(lambda i: data[i][0], range(len(data))),map(lambda i: data[i][1], range(len(data)))))
            # Add modified nucleotide field
            context_sequence = entry['data']['context']
            modified_nucleotide = context_sequence[len(context_sequence)//2].upper()
            entry['nucleotide'] = modified_nucleotide
            return entry
        
        dGFF = {"Heading":[],"Body":[]}
        if not path or not os.path.exists(path):
            raise ValueError(f"")
            
        data = self.read(path,"text",inlist=True,separator="\t",strip_symbol=" ")
        for i in range(len(data)):
            if data[i][:2]=="##":
                dGFF["Heading"].append(data[i])
            elif len(data[i])==9:
                record = get_entry(data[i])
                dGFF["Body"].append(record)
            else:
                continue
        dGFF['Body'].sort(key=lambda d: [d['strand'],int(d['start'])])
        # Return GFF dictionary
        if mode == "dictionary":
            return dGFF
        # Create and return GFF object
        oGFF = GFF(path=path, heading=dGFF['Heading'], records=dGFF['Body'])
        return oGFF
    
    # Working with sequence files in fasta and genbank formats
    def parse_seq(self,path,seq_format="fasta",reverse_complement=False,concatenate=False,reverse_contigs=False):
        oParser = Parser()
        return oParser.parse(path,seq_format=seq_format,reverse_complement=reverse_complement,concatenate=concatenate,reverse_contigs=reverse_contigs)
    
    def read_seq(self,path,seq_format="fasta",reverse_complement=False):
        oParser = Parser()
        return oParser.read(path,seq_format=seq_format,reverse_complement=reverse_complement)
    
    # Create new folder
    def new_folder(self,folder_name):
        try:
            os.mkdir(folder_name)
            return folder_name
        except:
            return None
    
    # copy text files
    def copy(self,inpath,outpath,
                data_format="text",     # text | binary
                dbkey='$db$',           # used with binary, main key
                splkey='$suppl$',       # used with binary, supplementary key
                    ):
        if not os.path.exists(inpath):
            raise ValueError(f"Path {path} does not exist!")
        if data_format.upper()=="TEXT":
            try:
                self.save(read(inpath,data_format),outpath,data_format)
            except:
                raise TypeError(f"File {path} cannot be copied!")
        elif data_format.upper()=="BINARY":
            try:
                data,suppl_data = read(inpath,data_format,dbkey=dbkey,splkey=splkey)
                self.save(data,outpath,data_format,dbkey=dbkey,splkey=splkey,suppl_data=suppl_data)
            except:
                raise TypeError(f"File {path} cannot be copied!")
        return outpath
    

##############################################################################################################
# Collection of parsers
class Parser:
    def __init__(self, path="", seq_format=""): # seq_format = 'genbank','fasta'
        self.path = path
        self.seq_format = seq_format
    
    # Interface functions read and parse to accept user's requests and identify data type
    def parse(self,path="",seq_format="",reverse_complement=False,concatenate=False,reverse_contigs=False):  # Seq format can be predefined by users, or identified by file extension
        path,seq_format = self._check_path_format(path,seq_format)
        if seq_format.upper() == 'GENBANK':
            oObj = self._parse_genbank(path)
        elif seq_format.upper() == 'FASTA':
            oObj = self._parse_fasta(path, concatenate=concatenate)
        elif seq_format.upper() == 'GFF':
            return self._parse_gff(path)
        else:
            raise TypeError(f"Parsing of {seq_format} files is not supported!")
        if reverse_contigs:
            oObj = oObj.reverse()
        if reverse_complement:
            oObj = oObj.reverse_complement()
        if concatenate:
            oObj = oObj.get_concatenated()
        elif len(oObj)==0:
            raise ValueError(f"No sequences were found in file {path}")
        return oObj
                
    def read(self,path="",seq_format="",reverse_complement=False):  # Seq format can be predefined by users, or identified by file extension
        oObj = self.parse(path,seq_format=seq_format,reverse_complement=reverse_complement)
        if len(oObj) > 1:
            msg(f"WARNING: file {path} contains more than one sequence!\nOnly first sequence is returned. Use 'parse' command to get all sequences of this file")
        return oObj[0]
            
    # Check input path and file format
    def _check_path_format(self,path="",seq_format=""):
        if not path:
            path = self.path
        if not seq_format:
            seq_format = self.seq_format
            if not seq_format:
                seq_format = self._define_seq_format(path)
        return path,seq_format
            
    # Seq formats are defined by file extensions
    def _define_seq_format(self,path):
        if path and path[path.rfind(".")+1:].upper() in ("GBK","GB","GBF"):
            return "GENBANK"
        if path and path[path.rfind(".")+1:].upper() in ("FASTA","FA","FST","FNA","FAA","FFN","FRN"):
            return "FASTA"
        return ""
        
    # Remove from text line Linux elements and quotes
    def format_line(self,line):
        return line.replace("\r","").replace("\"","'")
        
    # FASTA format parser
    def _parse_fasta(self, path, concatenate=True):
        # Define the path to fasta file
        fasta_file = path
    
        # Initialize variables to store record information
        oFASTA = Fasta(os.path.basename(path[:path.rfind(".")]))
        
        # Open and read FASTA file
        with open(fasta_file, "r") as file:
            lines = file.readlines()
    
            current_title = ""
            current_sequence_lines = []
    
            lnum = 0
            while lnum < len(lines):
                line = lines[lnum].strip()
                if not line:
                    lnum += 1
                    continue
    
                if line.startswith(">"):
                    # If we already have a record, append it
                    if current_title:
                        seq = "".join(current_sequence_lines)
                        record = SeqRecord(seq=seq, title=current_title)
                        oFASTA.append(record)
                    
                    # Start new record
                    current_title = line[1:].strip()
                    current_sequence_lines = []
                else:
                    current_sequence_lines.append(line)
    
                lnum += 1
    
            # Append the last record if present
            if current_title:
                seq = "".join(current_sequence_lines)
                record = SeqRecord(seq=seq, title=current_title)
                oFASTA.append(record)
    
        # Concatenate sequences with 50 N's in between
        if concatenate:
            oFASTA.concatenate()
            
        return oFASTA
        
    # GenBank format parser
    def _parse_genbank(self,path):
        # Define the path to GenBank file
        genbank_file = path
        
        # Create GBF as a collection of GBK records
        oGBF = GBF(os.path.basename(path[:path.rfind(".")]))
        # Open and read the GenBank file
        with open(genbank_file, "r") as file:
            # Initialize variables to store record information
            heading = []
            current_sequence = ""
            current_feature = None
            current_features = []
            lines = file.readlines()
            in_sequence = False
            in_features = False
            title = ""
            
            lnum = 0
            while lnum < len(lines):
                line = self.format_line(lines[lnum])
                # Capture heading lines
                if not in_sequence and not in_features:
                    heading.append(line.replace("\n",""))
                    if len(heading) == 1:
                        title = line[12:33].strip()
                        
                # Scip base count line
                if line.startswith("BASE COUNT"):
                    lnum += 1
                    continue
                
                # Start capturing sequence data
                if line.startswith("ORIGIN"):
                    in_sequence = True
                    in_features = False
                    current_sequence = ""
                    
                # End sequence data capture
                elif in_sequence and line.startswith("//"):
                    in_sequence = False
                    # Add new GBK record
                    oGBF.append(GBK(title,"\n".join(heading),SeqRecord(current_sequence,title),current_features))
                    current_features = []
                
                # Capture sequence lines
                elif in_sequence:
                    current_sequence += line[10:].replace(" ", "").replace("\n", "").upper()
                
                # Capture feature lines
                elif in_features:
                    if len(line) > 22 and line[21]=="/" and current_feature != None:
                        try:
                            key,value = line.strip().replace("\"","")[1:].split("=")
                        except:
                            lnum += 1
                            continue
                        # concatenate values in multiple lines
                        lnum += 1
                        line = lines[lnum]
                        while lnum < len(lines) and len(line) > 22 and line[5]==" " and line[21] != "/":
                            if key == "translation":
                                value += line.strip().replace("\"","")
                            else:
                                value += " " + line.strip().replace("\"","")
                            lnum += 1
                            line = lines[lnum]
                        current_feature.qualifiers[key] = [value.replace("\n","")]
                        continue
                    elif len(line) > 5 and line[5] != " ":
                        if current_feature:
                            current_features.append(current_feature)
                        feature_type = line[5:line.find(" ",5)]
                        strand = 1
                        line = line.replace("join(","").replace(")","")
                        if line.find("complement") > -1:
                            strand = -1
                            line = line.replace("complement(","")
                        exons = line[21:].strip().split(",")
                        current_feature = Feature(feature_type,strand,exons)
                        
                # Start capturing feature lines
                elif line.startswith("FEATURES"):
                    in_features = True
                
                # End feature data capture
                elif line.startswith("ORIGIN") or line.startswith("//"):
                    in_features = False
                    current_feature = None
                
                lnum += 1
        oGBF.concatenate()
        return oGBF
    
    # GFF file format parser
    def _parse_gff(self,path):
        # Define the path to GenBank file
        gff_file = path        
        # Create GFF as a collection of records representing each line in GFF file
        oGFF = GFF(os.path.basename(path[:path.rfind(".")]))
        # Open and read the GenBank file
        with open(gff_file, "r") as file:
            # Initialize variables to store record information
            heading = []
            in_features = False
            lines = file.readlines()
            lnum = 0
            while lnum < len(lines):
                line = self.format_line(lines[lnum])
                # Capture heading lines
                if line.startswith("##"):
                    heading.append(line)
                # Capture data lines as records
                elif len(line):
                    try:
                        entry = dict(zip(["genome","method","modtype","start","end","score","strand","para","data"],[s.strip() for s in line.split("\t")]))
                        data = list(map(lambda s: s.split("="), entry["data"].split(";")))
                        entry["data"] = dict(zip([data[i][0] for i in range(len(data))], [data[i][1] for i in range(len(data))]))
                        oGFF.append(Record(f"{entry['strand']}{entry['start']}..{entry['end']}",entry))
                    except:
                        raise ValueError(f"Line {line} cannot be parsed as a GFF record!")
                    
                else:
                    pass
                lnum += 1
        if heading:
            oGFF['Heading'] = "\n".join(heading)
        oGFF['Body'].sort(lambda d: [d.strand,int(d.start)])
        return oGFF
        
##############################################################################################################
class Fasta(Collection):
    def __init__(self,title="",spacer_length=50):
        Collection.__init__(self, title)
        self.description = self.title
        self.Seq = self.seq = None
        
    def __add__(self,other):
        if not isinstance(other,Fasta):
            raise TypeError("unsupported types of operands!")
        self.extend(other.get())
        self.concatenate()
        
    def __repr__(self):
        output = [f"{self.title} <{len(self.container)} record(s)>"]
        for oGBK in self:
            output.append(f"\t>{oSeq.title}\n\t{str(oSeq)[:20]}...")
        return "\n".join(output)
        
    def base_count(self,nucleotides=[],concatenated=False):
        if nucleotides == []:
            nucleotides = ['a','c','g','t']
        if concatenated:
            return " ".join([f"{nuc} {self.Seq.base_count(nuc)}" for nuc in nucleotides])        
        return "\n".join([" ".join([f"{nuc} {oSeq.base_count(nuc)}" for nuc in nucleotides]) for oSeq in self])
                                    
    def get_concatenated(self):
        if self.Seq==None:
            self.concatenate()
        return self.Seq
            
    def concatenate(self):
        # Concatenate sequences with 50 N's in between
        if len(self.container) > 1:
            sequences = [str(oSeq) for oSeq in self.get()]
            concatenated_sequence = self.spacer.join(sequences)
            self.Seq = self.seq = SeqRecord(concatenated_sequence,self.title)
        
        elif len(self.container):
            self.Seq = self.seq = self[0]
            
    def reverse_contigs(self):
        return self.reverse()
            
    def reverse_complement(self,reverse=False):
        # Create a copy
        seq_list = [oSeq.reverse_complement() for oSeq in self.get()]
        if reverse:
            seq_list.reverse()
        self.push(seq_list)
        self.concatenate()

    # return data as a formated text
    def format(self,concatenated_sequence=False):
        if concatenated_sequence:
            return self.Seq.format("fasta")
        return "\n".join([oSeq.format("fasta") for oSeq in self.get()])
    
    def copy(self,reverse_complement=False):
        oFasta = Fasta(self.title,len(self.spacer))
        oFasta.push(self.get())
        oFasta.concatenate()
        return oFasta
    
##############################################################################################################
class SeqRecord:
    def __init__(self,seq="",title="",moltype="DNA"):
        self.title = title
        self.description = self.title
        self.Seq = seq
        self.seq = self.Seq
        self.moltype = moltype
        
    def __len__(self):
        return len(self.Seq)
        
    def __str__(self):
        return self.Seq
        
    def __getitem__(self,index):
        if isinstance(index, slice):
            return SeqRecord(self.Seq[index.start:index.stop])
        return self.Seq[index]

    # Return GBK formated lines of sequence
    def _format_fasta(self,line_length = 100):
        fasta = [f">{self.title}"]
        fasta += [self.Seq[i:i + line_length].strip() if i <= len(self.Seq) - line_length else self.Seq[i:]
            for i in range(0, len(self.Seq), line_length)]
        return "\n".join(fasta)
        
    def _format_genbank(self,indend = 9,line_length = 66):
        seq = " ".join([self.Seq[i:i + 10] if i <= len(self.Seq) - 10 else self.Seq[i:] for i in range(0, len(self.Seq), 10)]).lower()
        seq = [seq[i:i + line_length].strip() if i <= len(seq) - line_length else seq[i:] for i in range(0, len(seq), line_length)]
        num = 1
        for i in range(len(seq)):
            length = len(seq[i].replace(" ",""))
            seq[i] = " "*(indend-len(str(num)))+str(num)+" " + seq[i]
            num += length
        return seq
        
    def format(self,seq_format="FASTA"):
        if seq_format.upper()=="FASTA":
            return self._format_fasta()
        elif seq_format.upper()=="GENBANK":
            return self._format_genbank()
            
    def base_count(self,nuc):
        return self.Seq.upper().count(nuc.upper())
            
    def reverse_complement(self,seq):
        # Create a translation table for nucleotides, including ambiguous ones.
        # This uses the str.maketrans() function to map each nucleotide to its complement.
        complement_table = str.maketrans(
            "ATGCRYWSKMBDHVNatgcrywskmbdhvn",
            "TACGYRWSMKVHDBNatgcyrwsmkvhdbn"
        )        
        # Translate the sequence using the translation table, then reverse it
        return seq.translate(complement_table)[::-1]    
    
    def translate(self,dna_seq="",strand=1):
        if not dna_seq and self.moltype=="DNA":
            dna_seq = self.Seq
        elif isinstance(dna_seq,SeqRecord) and dna_seq.moltype=="DNA":
            dna_seq = str(dna_seq.Seq)
        elif isinstance(dna_seq,str):
            pass
        else:
            raise TypeError(f"Object {dna_seq} cannot be translated!")
        if strand == -1:
            dna_seq = self.reverse_complement(dna_seq)
        # Bacterial codon table 11 (standard)
        codon_table = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        protein_seq = ''
        for i in range(0, len(dna_seq), 3):
            codon = dna_seq[i:i+3]
            # Translate the codon to an amino acid and append to the protein sequence
            protein_seq += codon_table.get(codon.upper(), 'X')  # 'X' for unknown codons
        return SeqRecord(protein_seq,self.title,"protein")
    
    def copy(self,reverse_complement=False):
        if reverse_complement:
            return SeqRecord(self.reverse_complement(self.Seq),self.title)
        return SeqRecord(self.Seq,self.title,self.moltype)
    
##############################################################################################################
class Location:
    def __init__(self,exons=["0..0"]):
        self.exons = list(map(lambda s: list(map(lambda v: int(v.replace(">","").replace("<","")), s.split(".."))), exons))
        #self.exons = [[int(v.replace(">", "").replace("<", "")) for v in s.split("..")] for s in exons]
        #self.exons = [ exons]
        self.start = self.exons[0][0]
        self.end = self.exons[-1][1]
        
    # Format GBK feature location
    def get_exons(self,start=0,complement=0):
        # 'start' will be added to start and locations
        # if 'complement' > 0, the valu means the total sequence length
        if not complement:
            return [f"{locus[0] + start}..{locus[1] + start}" for locus in self.exons]        
        return [f"{complement - locus[1] + start}..{complement - locus[0] + start}" for locus in self.exons]
            
    def format(self):
        exons = self.get_exons()
        if len(exons) > 1:
            return f"join({','.join(exons)})"
        return ','.join(exons)
        
    def copy(self,start=0,complement=0):
        return Location(self.get_exons(start=start,complement=complement))

##############################################################################################################
class GBF(Collection):
    def __init__(self,title="",spacer_length=50):
        Collection.__init__(self,title)
        self.description = self.title
        self.GBK = None
        self.spacer = "N" * spacer_length
        
    def __add__(self,other):
        if not isinstance(other,GBF):
            raise TypeError("unsupported types of operands!")
        self.extend(other.get())
        self.concatenate()
        
    def __repr__(self):
        output = [f"{self.title} <{len(self.container)} record(s)>"]
        for oGBK in self:
            output.append(f"\t>{oGBK.title}\n\t{str(oGBK.Seq)[:20]}...")
        return "\n".join(output)
        
    def base_count(self,nucleotides=[],concatenated=False):
        if nucleotides == []:
            nucleotides = ['a','c','g','t']
        if concatenated:
            return " ".join([f"{nuc} {self.GBK.Seq.base_count(nuc)}" for nuc in nucleotides])
        return "\n".join([" ".join([f"{nuc} {oGBK.Seq.base_count(nuc)}" for nuc in nucleotides]) for oGBK in self])
                                    
    def concatenate(self):
        # Concatenate sequences with 50 N's separators
        if len(self) > 1:
            sequences = [str(oGBK.Seq) for oGBK in self]
            concatenated_sequence = self.spacer.join(sequences)
            features = []
            contig_start = 0
            contig_counter = 1
            for i in range(len(sequences)):
                sequence = sequences[i]
                contig_end = contig_start + len(sequence)
                # Add contig features
                features.append([Feature("contig",1,[f"{contig_start}..{contig_end}"])]+self[i].get_features(contig_start))
                features[-1][0].qualifiers["note"] = [f"Contig_{contig_counter}"]
                contig_start = contig_end + 1
                contig_counter += 1
            self.GBK = GBK(self.title,"",SeqRecord(concatenated_sequence,self.title),features)
        
        elif len(self) == 1:
            self.GBK = self[0]
            
    def get_concatenated(self):
        if self.GBK==None:
            self.concatenate()
        return self.GBK
            
    def reverse_complement(self,reverse=False):
        # Create a copy
        gbk_list = [oGBK.reverse_complement() for oGBK in self]
        if reverse:
            gbk_list.reverse()
        self.push(gbk_list)
        self.concatenate()

    def reverse_contigs(self):
        return self.reverse()
            
    # return data as a formated text
    def format(self,seq_format="GENBANK",concatenated_sequence=False):
        if concatenated_sequence:
            if seq_format.upper()=="GENBANK":
                return self.GBK._format_genbank()
            elif seq_format.upper()=="FASTA":
                return self.GBK._format_fasta()
            else:
                msg("Wrong file format %s!" % seq_format)
                return ""
        if seq_format.upper()=="GENBANK":
            #return "\n".join(list(map(lambda oGBK: oGBK._format_genbank(), self.get())))
            return "\n".join([oGBK._format_genbank() for oGBK in self.get()])
        elif seq_format.upper()=="FASTA":
            #return "\n".join(list(map(lambda oGBK: oGBK._format_fasta(), self.get())))
            return "\n".join([oGBK._format_fasta() for oGBK in self.get()])
        else:
            msg("Wrong file format %s!" % seq_format)
            return ""
    
    def copy(self,reverse_complement=False):
        oGBF = GBF(self.title,len(self.spacer))
        oGBF.push(self.get())
        oGBF.concatenate()
        return oGBF

##############################################################################################################
class GBK:
    def __init__(self,title="",heading="",oSeq=None,features=[]):
        self.title = title
        self.description = self.title
        self.heading = heading
        self.description = self.title
        self.accession = self.title
        self.Seq = self.seq = oSeq
        self.features = features
                
    def __len__(self):
        return len(self.Seq)

    def __repr__(self):
        return f"Title: {self.title}\nSeq. Length: {len(self.Seq)}\nNum. Features: {len(self.features)}"
        
    def __str__(self):
        return self.__repr__()
        
    def __getitem__(self,index):
        if isinstance(index, slice):
            oCopy = self.copy()
            # Set heading and title
            oCopy.heading = ""
            oCopy.title = self.title+"slice"
            oCopy.description = oCopy.title
            # Seq concatenated sequence
            oCopy.Seq += self.Seq[index.start:index.end]
            # Add features from the second GBK after an adjastment of their locations
            oCopy.features += [
                ft.copy(start=-index.start) 
                for ft in other.features 
                if ft.location.start >= index.start and ft.location.end <= index.stop
            ]
            return oCopy
        else:
            return self.Seq[index]
        
    def __add__(self,other):
        if not isinstance(other, GBK):
            raise TypeError("Both operands must be 'GBK' objects")
        # Create a copy
        oCopy = self.copy()
        # Set heading and title
        oCopy.heading = ""
        oCopy.title = self.title+"_concat"
        oCopy.description = oCopy.title
        # Seq concatenated sequence
        oCopy.Seq += other.Seq
        oCopy.seq = oCopy.Seq
        # Add features from the second GBK after an adjastment of their locations
        oCopy.features += [ft.copy(start=len(str(self.Seq))) for ft in other.features]
        return oCopy
    
    # Simmilar to addition, but separate sequences with N's
    def concatenate(self,other,separator_length=50):
        if not isinstance(other, GBK):
            raise TypeError("Both operands must be 'GBK' objects")
        # Create a copy
        oCopy = self.copy()
        # Set heading and title
        oCopy.heading = ""
        oCopy.title = self.title+"_concat"
        oCopy.description = oCopy.title
        # Seq concatenated sequence
        oCopy.Seq += "N"*separator_length + other.Seq
        oCopy.seq = oCopy.Seq
        # Add features from the second GBK after an adjastment of their locations
        oCopy.features += [ft.copy(start=len(str(self.Seq)))+separator_length for ft in other.features]
        return oCopy
    
    def rearrange(self,moltype="CDS",tag="",gene="",product="",strand=1):
        shift = 100
        # Create a copy
        oCopy = self.copy()
        # List of features
        #features = list(map(lambda ft: ft.copy(), oCopy.features))
        features = [ft.copy() for ft in oCopy.features]
        if features and moltype:
            features = [ft for ft in features if ft.type == moltype]        
        if features and tag:
            features = [ft for ft in features if 'locus_tag' in ft.qualifiers and ft.qualifiers['locus_tag'] and ft.qualifiers['locus_tag'][0] == tag]        
        if features and gene:
            features = [ft for ft in features if 'gene' in ft.qualifiers and ft.qualifiers['gene'] and ft.qualifiers['gene'][0] == gene]        
        if features and product:
            features = [ft for ft in features if 'product' in ft.qualifiers and ft.qualifiers['product'] and ft.qualifiers['product'][0] == product]
        if not len(feature):
            return None
            
        ft = features[0]
        reverse_complement = 0
        if strand != ft.strand:
            reverse_complement = len(self.Seq)
        if reverse_complement:
            if ft.location.end+shift >= len(self.Seq):
                shift = len(self.Seq)-ft.location.end-1
            oCopy = oCopy[ft.location.end+shift:]+oCopy[:ft.location.end+shift]
            return oCopy.reverse_complement()
        if ft.location.start-shift < 0:
            shift = ft.location.start
        return oCopy[ft.location.start-shift:]+oCopy[:ft.location.start-shift]
        
    def get_features(self,increment=0):
        return [ft.copy(increment) for ft in self.features]
        
    def reverse_complement(self):
        # Create a copy
        return self.copy(True)

    # return data as a formated text
    def format(self,seq_format="GENBANK"):
        if seq_format.upper()=="GENBANK":
            return self._format_genbank()
        elif seq_format.upper()=="FASTA":
            return self._format_fasta()
        else:
            msg("Wrong file format %s!" % seq_format)
            return ""
            
    # Get features in fasta format
    def features2fasta(self,moltype="CDS",seqtype="PROT"):
        if moltype:
            features = [ft for ft in self.features if ft.type==moltype]
        else:
            features = self.features
        if moltype.upper() == "DNA":
            return "\n".join([f">{oFT.get_feature_title()}\n{str(self.Seq[oFT.start - 1:oFT.end])}" for oFT in self])
        elif moltype.upper() in ("PROT","PROTEIN","AMC"):
            fasta = []
            for oFT in features:
                if 'translation' in oFT.qualifiers:
                    fasta.append(f">{oFT.get_feature_title()}\n{oFT.qualifiers['translation'][0]}")
                else:
                    fasta.append(f">{oFT.get_feature_title()}\n{self.Seq.translate(str(self.Seq[oFT.start-1:oFT.end]),oFT.strand)}")
            return "\n".join(fasta)
        else:
            raise ValueError(f"Sequence type {moltype} is not supported!")
    
    def _format_genbank(self):
        # Get first GBK line
        gbk = [self._get_first_line()]
        # Format heading
        heading = self.heading.split("\n")
        if len(heading) > 1:
            gbk += heading[1:]
        # Format features and sequence
        gbk += (
            sum([ft.format() for ft in self.features], []) +
            ["BASE COUNT   " + " ".join([f"{nuc} {self.Seq.base_count(nuc)}" for nuc in ['a', 'c', 'g', 't']]), "ORIGIN"] +
            self.Seq.format("genbank") +
            ["//", ""]
        )            
        return "\n".join(gbk)
        
    def _format_fasta(self):
        return self.Seq.format("fasta")
        
    def _get_first_line(self):
        # Format title
        title = self.title
        if len(title) > 20:
            title = title[:17]+"..."
        # Get the current date
        current_date = datetime.now()        
        # Format the date as "UNK DD-MMM-YYYY"
        formatted_date = "UNK " + current_date.strftime("%d-%b-%Y").upper()
        # Format first line of GBK file
        return ("LOCUS       %s%s%d bp%sDNA              %s" %
                (title," "*(21-len(title)),len(str(self.Seq))," "*(11-len(self.Seq)),formatted_date))
                
    def copy(self, reverse_complement=False):
        if reverse_complement:
            return GBK(
                self.title,
                self.heading,
                self.seq.copy(reverse_complement=True),
                [ft.copy(complement=len(self.Seq)) for ft in self.features]
            )
        
        return GBK(
            self.title,
            self.heading,
            self.Seq.copy(),
            [ft.copy() for ft in self.features]
        )

##############################################################################################################
class Feature:
    def __init__(self,ftype="",strand=0,exons=["0..0"]):
        self.type = ftype
        self.strand = strand
        self.location = Location(exons)
        self.qualifiers = {}
    
    def __repr__(self):
        return f"Type: {self.type}\tLocation: {self.location.start}..{self.location.end}; strand: {self.strand}"
        
    def __str__(self):
        return self.__repr__()
        
    # Generate feature title
    def _get_feature_title(self):
        feature_title = self.type+":"
        if 'locus_tag' in self.qualifiers:
            feature_title += f" [{self.qualifiers['locus_tag'][0]}]"
        if 'gene' in self.qualifiers:
            feature_title += f" ({self.qualifiers['gene'][0]})"
        if 'product' in self.qualifiers:
            feature_title += f" {self.qualifiers['product'][0]}"
        return feature_title

    # Return GBK formated qualifier data
    def _format_qualifier(self,key):
        output = []
        if key not in list(self.qualifiers.keys()):
            return output
        for i in range(len(self.qualifiers[key])):
            output += self._format_feature_text(key,self.qualifiers[key][i])
        return output
        
    # Return GBK formated lines of feature data
    def _format_feature_text(self,key,value):
        field_length = 80
        indend = 21
        length = field_length-indend
        output = [" "*indend]
        line = ("/"+key+"=\""+value+"\"").split(" ")
        while line:
            while len(output[-1]) < field_length:
                text_element = line[0]
                if len(text_element) > length:
                    line[0] = line[0][field_length-len(output[-1])-1:]
                    text_element = text_element[:field_length-len(output[-1])]
                elif len(output[-1])+len(text_element) > field_length:
                    break
                else:
                    line = line[1:]
                output[-1] += text_element+" "
                if not line:
                    break
            output.append(" "*indend)
        output = [s for s in output if len(s.replace(" ","")) > 1]
        return output
    
    # Format GBK feature location
    def _format_location(self):
        # Check strand of the feature
        if self.strand == 1:
            return self.location.format()
        else:
            return f"complement({self.location.format()})"
            
    # Return GBK formated feature
    def format(self):
        # Format feature heading
        output = [" "*5 + self.type + " "*(16-len(self.type)) + self._format_location()]
        # Add qualifiers
        for key in list(self.qualifiers.keys()):
            output += self._format_qualifier(key)
        return output
        
    # Return a unique identifier
    def get_tag(self,index=0):
        if 'locus_tag' in self.qualifiers:
            return self.qualifiers['locus_tag'][0]
        return f"{self.type}_{index}"
        
    def copy(self,start=0,complement=0): 
        # 'start' will be added to start and locations
        # if 'complement' > 0, the valu means the total sequence length
        oCopy = Feature(self.type, self.strand, [exon for exon in self.location.get_exons(start=start, complement=complement)])
        oCopy.location = self.location.copy(start=start,complement=complement)
        oCopy.qualifiers = copy.deepcopy(self.qualifiers)
        return oCopy
        
##############################################################################################################
class GFF(Collection):
    def __init__(self, path="", heading="", records=[]):
        Collection.__init__(self, path)
        self.path = self.title
        self.heading = self.Heading = heading
        self.records = self.body = self.Body = Container()
        for record in records:
            self.records.append(GFF_record(**record))
        
    def __str__(self):
        return f"File: {self.path}\n{len(self.body)} records"
        
    def copy(self):
        oGffCopy = GFF(path=self.title)
        oGffCopy.heading = self.heading
        oGffCopy.body = self.body.copy()
        return oGffCopy
        
##############################################################################################################
class Record:
    def __init__(self, **attributes):

        # Parse attributes
        for key, value in list(attributes.items()):
            if isinstance(value, dict):
                setattr(self, key, Record(**value))
            else:
                setattr(self, key, value)
                    
    def __getitem__(self,attr):
        if isinstance(attr, str) and hasattr(self,attr):
            return eval(compile(f"self.{index_start}", "<string>", "eval"))
        raise ValueError(f"Object {type(self)} has no attribute {str(attr)}!")
    
    def __setitem__(self,attr,value):
        if isinstance(attr, str) and hasattr(self,attr):
            setattr(self, attr, value)
        raise ValueError(f"Object {type(self)} has no attribute {str(attr)}!")
        
##############################################################################################################

class GFF_record(Record):
    def __init__(self,  **attributes):
        Record.__init__(self, **attributes)

##############################################################################################################
if __name__ == "__main__":
    oSeqIO = IO()
    path = os.path.join("..","input","S.aureus_150.gbk")
    oGBK = oSeqIO.read(path,"genbank")
    oSeqIO.save(oGBK.format("genbank"),"S.aureus_150.gbk")
    oSeqIO.save(oGBK.format("fasta"),"S.aureus_150.fa")
    

# =============================================================================
# BLAST wrapper, records, and parsers
# =============================================================================
##############################################################################################################
class container(Collection):
    """Backward-compatible BLAST container.

    The old BLAST module implemented a second list container with methods that
    largely duplicated :class:`Collection`.  This adapter keeps the legacy
    class name and BLAST-facing helper methods while reusing the shared
    Collection implementation.
    """

    def __init__(self, title: str = "") -> None:
        super().__init__(title)

    def get(self):
        """Return copies when possible, matching the historical BLAST behavior."""
        records = []
        for record in self.container:
            try:
                records.append(record.copy())
            except (AttributeError, TypeError):
                records.append(copy.deepcopy(record))
        return records

    def __iter__(self):
        return iter(self.get())

    def format_title(self, title: str, length: int, space: int) -> list[str]:
        """Split a title into display lines of approximately ``space`` chars."""
        words = f"{title} ({length} bp)".split()
        lines = [""]
        for word in words:
            candidate = f"{lines[-1]}{word} "
            if lines[-1] and len(candidate) > space:
                lines.append(f"{word} ")
            else:
                lines[-1] = candidate
        return lines

    def sort_container(self, sort_fn=None, flg_reverse: bool = False) -> None:
        """Legacy alias for in-place sorting of contained objects."""
        self.container.sort(key=sort_fn, reverse=flg_reverse)

    def remove_all(self) -> None:
        """Legacy alias used by :class:`BLAST` before each execution."""
        self.container.clear()
                    
##############################################################################################################
class BLAST(container):
    ###################################################
    ####  program in blastn,blastp,bl2seq,bl2seqp
    ####  path - executables
    ####  ref - path, sequence, blast database or fasta
    ####        file to be converted to blastdb
    ####  query - path or sequence
    ###################################################
    def __init__(self,seqtype="dna",binpath="",source_path=""):
        self.seqtype = seqtype
        self.path = binpath
        self.source_path = source_path
        self.query = ""
        self.sbjct = ""
        self.cline = ""
        container.__init__(self)
    
    def _set_cline(self,program="blast"):
        if self.seqtype == "dna":
            algorithm = "blastn"
            flg_prot = "F"
            dbtype = "nucl"
        elif self.seqtype == "protein":
            algorithm = "blastp"
            flg_prot = "T"
            dbtype = "prot"
        else:
            print("\nSequences type was not specified!\n")
            return
        if not self.query or not self.sbjct:
            print(f"\nEither query {self.query} or subject {self.sbjct} are wrong!\n")
        query = os.path.join(self.source_path,self.query)
        sbjct = os.path.join(self.source_path,self.sbjct)
        # On Windows
        if sys.platform == "win32":
            if program == "formatdb":
                self.cline = f"{os.path.join(self.path, 'formatdb')} -i {query} -p {flg_prot} -n {sbjct}"
            elif program == "blast":
                default = "-G 6 -E 2 -F F -q -2 -r 1 -e 1.0"
                self.cline = f"{os.path.join(self.path, 'blastall')} -p {algorithm} -i {query} -d {sbjct} {default}"
        # On Linux
        elif sys.platform.startswith("linux"):  # Corrected platform check for Linux
            if program == "formatdb":
                self.cline = f"{os.path.join(self.path, 'makeblastdb')} -in {query} -dbtype {dbtype} -out {sbjct}"
            elif program == "blast":
                if algorithm == "blastn":
                    default = "-gapopen 6 -gapextend 2 -evalue 1.0 -dust no -soft_masking false -penalty -2 -reward 1"
                elif algorithm == "blastp":
                    default = "-gapopen 6 -gapextend 2 -evalue 1.0 -seg no -matrix BLOSUM62"
                else:
                    raise ValueError(f"Unsupported BLAST algorithm: {algorithm}")
            
                self.cline = f"{os.path.join(self.path, algorithm)} -query {query} -db {sbjct} {default}"                
                          
        else:
            raise ValueError(f"Unsupported platform: {sys.platform}")
            
    def execute(self,query="",sbjct="",print_output=False):
        self.remove_all()
        self.query = query
        self.sbjct = sbjct
        self._set_cline()
        output = self.process(self.cline)
        if not output:
            return False
        if print_output:
            print("blast output:\n",output)
            
        oParser = parser("blast",output)
        self.container = oParser()
        return True
    
    def create_db(self,fasta_file,dbname):
        self.query = fasta_file
        self.sbjct = dbname
        self._set_cline("formatdb")
        self.process(self.cline)
    
    def process(self, cline, flg_wait=False):
        # Current working directory
        cwd = ""
        # Determine if we need to run the command in the shell based on the platform
        use_shell = sys.platform == "win32"
        
        #print(use_shell,isinstance(cline, str))
        # Ensure `cline` is a list if `shell=False` (Linux)
        if not use_shell and isinstance(cline, str):
            # If cline is a string, we split it into a list for Linux
            cline = cline.split(" ")
            cwd = os.getcwd()
            bin_directory = os.path.dirname(cline[0])
            command_name = os.path.basename(cline[0])
            os.chdir(bin_directory)
            cline[0] = command_name
        
        try:
             # Create the subprocess with the appropriate shell setting for Windows/Linux
            process = subprocess.Popen(cline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=use_shell, text=True)
    
            # Wait for the process to complete if requested
            if flg_wait:
                process.wait()
    
            # Read the command's output and error output
            output, errors = process.communicate()
    
            # Check if there were any errors and print them
            if errors:
                print(f"Error occurred: {errors}")
            
            # Close the stdin after sending input (if any)
            process.stdin.close()
    
            # Return to the working directory
            if cwd:
                os.chdir(cwd)

            # Return the output from stdout
            return output
    
        except Exception as e:
            print(f"Exception occurred while running command: {e}")
            return None
            
    def tostring(self,e_threshold=None):
        query_summary = header = output = ""
        sbjct_summary = {}
        for record in self:
            query_name = record.title
            header += "Query: " + query_name + "\n\n"
            output += "Query: " + query_name + "\n\n"
            for i in range(len(record.descriptions)):
                info = record.descriptions[i]
                if e_threshold is not None and e_threshold < info.e:
                    continue
                alignment = record[i]
                header += "\tSubject: " + alignment.source + ("\t%f\t%f" % (info.score,info.e)) + "\n"
                if alignment.source not in sbjct_summary:
                    sbjct_summary[alignment.source] = []
                sbjct_summary[alignment.source].append(query_name)
                output += ("\tSubject: "+alignment.title+"\n\n")
                for hsp in alignment:
                    if hsp.positives == None:
                        output += ("\tScore = %f, E-value = %f,\n\tIdentities = %f\n\n" %
                            (hsp.score,hsp.expect,hsp.identities))
                    else:
                        output += ("\tScore = %f, E-value = %f,\n\tIdentities = %f, Positives = %f\n\n" %
                            (hsp.score,hsp.expect,hsp.identities,hsp.positives))
                    point = 0
                    q_gaps = s_gaps = 0
                    while point < hsp.alignment_length:
                        end = point+60
                        if end > hsp.alignment_length:
                            end = hsp.alignment_length
                        q_substr = hsp.query[point:end].upper()
                        s_substr = hsp.sbjct[point:end].upper()
                        qs = q_gaps
                        ss = s_gaps
                        q_gaps += q_substr.count("-")
                        s_gaps += s_substr.count("-")
                        output += "%g\t\t%s\t\t%g\n" % (hsp.query_start+point-qs,q_substr,hsp.query_start+end-1-q_gaps)
                        output += "%g\t\t%s\t\t%g\n\n" % (hsp.sbjct_start+point-ss,s_substr,hsp.sbjct_start+end-1-s_gaps)
                        point += 60
                output += "\n"
            header += "\n"
        header += "#"*60
        if sbjct_summary:
            query_summary += "Subject summary:\n"
            sbjct_summary = sbjct_summary.items()
            if len(sbjct_summary) > 1:
                sbjct_summary.sort(key=lambda ls: ls[1])
            for item in sbjct_summary:
                query_summary += "\t".join(["",str(len(item[1])),item[0]])+"\n"
        return "\n\n".join([query_summary,header,output])
    
    def svg(self,X=25,Y=25,width=900,height=200,flg_finish=True):
        svg = []
        title_width=200
        title_height = 50
        title_start = width-title_width+5
        if not len(self):
            return ""
        for record in self:
            # Find the longest sequence
            max_length = record.query_length
            for sbjct in record:
                if sbjct.sbjct_length > max_length:
                    max_length = sbjct.sbjct_length
            query_width = title_width+float(width-title_width)*record.query_length/max_length
            query_svg = record.svg(X,Y,self.query_genemap,query_width,title_height,title_width,title_start,False)
            for sbjct in record:
                alignment_summary = sbjct.summarize()
                svg.append("<text x=\"%d\" y=\"%d\">Summarized score = %d; Best expectation = %f</text>" % 
                    (X,Y,alignment_summary.score,alignment_summary.expect))
                Y += 20
                if self.program == "bl2seq" and self.seqtype in ("dna","codon") and len(record)==1:
                    sbjct_width = title_width+float(width-title_width)*sbjct.sbjct_length/max_length
                    span = height-2*title_height
                    sbjct_svg = sbjct.svg(X,Y+span,self.sbjct_genemap,sbjct_width,title_height,title_width,title_start,False)
                    hsp_svg = sbjct.svg_hsps(float(width-title_width)/max_length,X,Y+title_height,width-title_width,span-title_height,False)
                else:
                    pass
                Y += height
                svg.extend([query_svg,hsp_svg,sbjct_svg])
        if flg_finish:
            svg.insert(0,"<svg xmlns=\"http://www.w3.org/2000/svg\" viewbox=\"0 0 %d %d\">" % (width,Y+600))
            svg.append("</svg>")
        return "\n".join(svg)

    def get_record(self,title):
        for record in self.container:
            if record.title == title:
                return record.copy()
        return None
        
    def get_top_alignment(self):
        if len(self.container) and len(self.container[0].container):
            if len(self.container) > 1:
                self.sort_container(sort_fn=lambda aln: aln.score, flg_reverse=True)
            record = self.container[0]
            if len(record.container[0]) > 1:
                record.container[0].sort_container(sort_fn=lambda obj: obj.identities, flg_reverse=True)
            hsps = list(map(lambda hsp: hsp.copy(), record.container[0]))
            return [record.container[0].e,record.container[0].score,record.container[0].title,hsps]
        return None
    
    def get_matches(self, query_length, mismatches=0):
        mismatches = int(mismatches)
        hsps = []
        for record in self.container:
            for alignment in record.container:
                hsps.extend(
                    rec.copy() for rec in alignment.container
                    if query_length - rec.identities + rec.query.count("-") + rec.sbjct.count("-") <= mismatches
                )                
        return hsps
        
##############################################################################################################
class blast_record(container):
    def __init__(self,query_length,title=""):
        self.query_length = query_length
        self.title = title
        self.top_score = 0
        self.summerized_score = 0.0
        self.descriptions = []
        container.__init__(self)
        
    def copy(self):
        record = blast_record(self.query_length,self.title)
        for description in self.descriptions:
            record.add_description(description.title,description.score,description.e)
        for alignment in self.container:
            record.add_alignment(alignment.copy())
        return record
        
    def add_description(self,title,score,e):
        description = blast_description(title,score,e)
        self.descriptions.append(description)
    
    def add_alignment(self,alignment):
        self.container.append(alignment)
    
    def svg(self,X=5,Y=25,genemap=[],width=800,height=50,title_width=100,title_start=0,flg_finish=True):
        svg = []
        svg.append("<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%f\" />" %
            (X,Y+height/2,X+width-title_width,Y+height/2,"red",3.0))
        # GENE MAP
        c = float(width-title_width)/self.query_length
        for gene in genemap:
            try:
                lb,rb = list(map(lambda s: int(s),gene.split("-")))
            except:
                lb,rb = list(map(lambda s: int(s),gene.split("..")))
            if lb < 0:
                lb = 0
            if rb > self.query_length:
                rb = self.query_length
            if (genemap[gene]['remark'].find("hypothetical") > -1 or
                genemap[gene]['name'].find("hypothetical") > -1 or
                genemap[gene]['remark'].find("unknown") > -1 or
                genemap[gene]['name'].find("unknown") > -1):
                color = "grey"
            else:
                color = "green"
            bar_height = height/5
            shift = height/7
            if genemap[gene]['direction'] == "rev":
                shift = height-bar_height-height/7
            svg.append("<rect x=\"%f\" y=\"%f\" width=\"%f\" height=\"%f\" style=\"fill:%s;stroke:%s\" />" %
                (X+c*lb,Y+shift,c*(rb-lb),bar_height,color,"grey"))
        # TITLE
        title = self.format_title(self.title,self.query_length,title_width/5)
        for subtitle in title:
            svg.append("<text x=\"%d\" y=\"%d\">%s</text>" % (X+title_start,Y+5,subtitle))
            Y += 15
        if flg_finish:
            svg.insert(0,"<svg xmlns=\"http://www.w3.org/2000/svg\" viewbox=\"0 0 %d %d\">" % (width,height))
            svg.append("</svg>")
        return "\n".join(svg)
    
    def sort(self):
        if len(self.container) > 1:
            self.container.sort(key=lambda aln: [aln.e,-aln.score], reverse=True)

##############################################################################################################
class blast_description:
    def __init__(self,title,score,e):
        self.title = title
        self.score = score
        self.e = e
        
##############################################################################################################
class blast_alignment(container):
    def __init__(self,sbjct_length=0,title="",score=0,e=None,hsps=[]):
        self.sbjct_length = sbjct_length
        self.title = title
        self.source = self.parse_title()
        self.score = score
        self.e = e
        container.__init__(self)
        self.container.extend(hsps)

    def copy(self):
        hsps = []
        for hsp in self.container:
            hsps.append(hsp.copy())
        alignment = blast_alignment(self.sbjct_length,self.title,self.score,self.e,hsps)
        return alignment
    
    def __add__(self,other=None):
        new_alignment = self.copy()
        if not other:
            return new_alignment
        new_alignment.title += "; "+other.title
        new_alignment.source += "; "+other.source
        new_alignment.score += other.score
        new_alignment.e = min(new_alignment.e,other.e)
        new_alignment.hsps.extend(other.hsps)
        return new_alignment

    def summarize(self):
        if not len(self.container):
            return None
        summarized_alignment = self.container[0].copy()
        for i in range(1,len(self.container)):
            summarized_alignment += self.container[i]
        return summarized_alignment
    
    def get_score(self):
        if not self.container:
            return None
        score = 0
        for hsp in self.container:
            score += hsp.score
        return score
    
    def get_expect(self):
        if not self.container:
            return None
        if len(self.container) == 1:
            return self.container[0].expect
        expect = self.container[0].expect
        for i in range(len(self.container),1):
            if self.container[i].expect < expect:
                expect = self.container[i].expect
        return expect
    
    def svg(self,X=5,Y=25,genemap=[],width=800,height=50,title_width=100,title_start=0,flg_finish=True):
        svg = []
        svg.append("<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" fill=\"none\" stroke=\"%s\" stroke-width=\"%f\" />" %
            (X,Y+height/2,X+width-title_width,Y+height/2,"red",3.0))
        # GENE MAP
        c = float(width-title_width)/self.sbjct_length
        for gene in genemap:
            try:
                lb,rb = list(map(lambda s: int(s),gene.split("-")))
            except:
                lb,rb = list(map(lambda s: int(s),gene.split("..")))
            if lb < 0:
                lb = 0
            if rb > self.sbjct_length:
                rb = self.sbjct_length
            if (genemap[gene]['remark'].find("hypothetical") > -1 or
                genemap[gene]['name'].find("hypothetical") > -1 or
                genemap[gene]['remark'].find("unknown") > -1 or
                genemap[gene]['name'].find("unknown") > -1):
                color = "grey"
            else:
                color = "green"
            bar_height = height/5
            shift = height/7
            if genemap[gene]['direction'] == "rev":
                shift = height-bar_height-height/7
            svg.append("<rect x=\"%f\" y=\"%f\" width=\"%f\" height=\"%f\" style=\"fill:%s;stroke:%s\" />" %
                (X+c*lb,Y+shift,c*(rb-lb),bar_height,color,"grey"))
        # TITLE
        title = self.format_title(self.title,self.sbjct_length,title_width/5)
        for subtitle in title:
            svg.append("<text x=\"%d\" y=\"%d\">%s</text>" % (X+title_start,Y+5,subtitle))
            Y += 15
        if flg_finish:
            svg.insert(0,"<svg xmlns=\"http://www.w3.org/2000/svg\" viewbox=\"0 0 %d %d\">" % (width,height))
            svg.append("</svg>")
        return "\n".join(svg)
    
    def svg_hsps(self,c,X=5,Y=25,width=800,height=150,flg_finish=True):
        svg = []
        for hsp in self:
            if not hsp.strand or hsp.strand=="Plus/Plus":
                svg.append("<path d=\"M%f,%dL%f,%dL%f,%dL%f,%dZ\" style=\"fill:%s;stroke:%s;opacity:%f\" />" %
                    (X+c*(hsp.query_start-1),Y,X+c*(hsp.query_end-1),Y,
                    X+c*(hsp.sbjct_end-1),Y+height,X+c*(hsp.sbjct_start-1),Y+height,
                    "blue","grey",80.0))
            elif hsp.strand=="Plus/Minus":
                svg.append("<path d=\"M%f,%dL%f,%dL%f,%dL%f,%dZ\" style=\"fill:%s;stroke:%s;opacity:%f\" />" %
                    (X+c*(hsp.query_start-1),Y,X+c*(hsp.query_end-1),Y,
                    X+c*(hsp.sbjct_start-1),Y+height,X+c*(hsp.sbjct_end-1),Y+height,
                    "blue","grey",80.0))
        if flg_finish:
            svg.insert(0,"<svg xmlns=\"http://www.w3.org/2000/svg\" viewbox=\"0 0 %d %d\">" % (width,height))
            svg.append("</svg>")
        return "\n".join(svg)
    
    def parse_title(self):
        if not self.title:
            return ""
        p = self.title.find("; from ")
        if p == -1:
            return self.title
        source = self.title.replace(", complete sequence.","")
        return source[p+7:]

##############################################################################################################
class blast_hsp:
    def __init__(self,score,e,aln_length,identities,positives,gaps,strand,qlb,slb,qrb,srb,query,sbjct,hits):
        self.score = score
        self.expect = e
        self.alignment_length = int(aln_length)
        self.identities = identities
        self.positives = positives
        self.gaps = gaps
        self.strand = strand
        self.query_start = qlb
        self.sbjct_start = slb
        self.query_end = qrb
        self.sbjct_end = srb
        self.query = query
        self.sbjct = sbjct
        self.match = hits
    
    def __str__(self):
        return "Query [%d..%d]; Sbjct [%d..%d]" % (self.query_start,self.query_end,self.sbjct_start,self.sbjct_end)
    
    def __repr__(self):
        return "Query [%d..%d]; Sbjct [%d..%d]" % (self.query_start,self.query_end,self.sbjct_start,self.sbjct_end)
    
    def measure_distance(self,location,shift=0,flg_abs=True):
        if self.strand == "Plus/Plus":
            dist = self.sbjct_start + int(shift) - int(location)
        else:
            dist = self.sbjct_end - int(shift) - int(location)
        if flg_abs:
            return abs(dist)
        return dist
    
    def get_strand(self):
        if self.strand == "Plus/Plus":
            return "+"
        return "-"
    
    def copy(self):
        hsp = blast_hsp(self.score,
                            self.expect,
                            self.alignment_length,
                            self.identities,
                            self.positives,
                            self.gaps,
                            self.strand,
                            self.query_start,
                            self.sbjct_start,
                            self.query_end,
                            self.sbjct_end,
                            self.query,
                            self.sbjct,
                            self.match)
        return hsp

    def __add__(self,other=None):
        new_hsp = self.copy()
        if not other:
            return new_hsp
        new_hsp.score += other.score
        new_hsp.expect = min(new_hsp.expect,other.expect)
        if other.identities:
            new_hsp.identities += other.identities
        if other.positives:
            new_hsp.positives += other.positives
        if other.gaps:
            new_hsp.gaps += other.gaps
        new_hsp.strand = ""
        new_hsp.query_start = min(new_hsp.query_start,other.query_start)
        new_hsp.query_stop = max(new_hsp.query_end,other.query_end)
        new_hsp.sbjct_start = min(new_hsp.sbjct_start,other.sbjct_start)
        new_hsp.sbjct_stop = max(new_hsp.sbjct_end,other.sbjct_end)
        new_hsp.query = ""
        new_hsp.sbjct = ""
        new_hsp.match = ""
        return new_hsp

##############################################################################################################
class parser:
    def __init__(self, program, raw_text):
        # Decode the text if it's a bytes object in Python 3
        if isinstance(raw_text, bytes):
            try:
                raw_text = raw_text.decode('utf-8')
            except UnicodeDecodeError as e:
                print(f"Error decoding raw_text: {e}")
        
        # Platform-specific parser instantiation
        if sys.platform == "win32":
            self.oParser = win_parser(program, raw_text)
        elif sys.platform.startswith("linux"):  # Updated for modern Python versions
            self.oParser = linux_parser(program, raw_text)
        else:
            raise ValueError(f"Unsupported platform: {sys.platform}")

    def __call__(self):
        return self.oParser._parse()     

##############################################################################################################
class sys_parser:
    def __init__(self,program,raw_text):
        self.program = program
        self.raw_text = raw_text
        self.dataset = []
        
    def _parse(self):
        self.raw_text = self.raw_text.replace("\r","")
        q = self.raw_text.find("Query= ")
        if q == -1:
            return
        while q != -1:
            t = self.raw_text.find("Query= ",q+1)
            title = self.raw_text[q+7:self.raw_text.find("\n",q)]
            query_length = int(self.raw_text[self.raw_text.find("         (",q)+10:self.raw_text.find(" letters)\n",q)].replace(",",""))
            record = blast_record(query_length,title)
            a = self.raw_text.find("Sequences producing significant alignments",q,t)
            if a == -1:
                q = self.raw_text.find("Query= ",q+1)
                continue
            else:
                a = self.raw_text.find("\n\n",a)+2
            b = self.raw_text.find(">",a)-2
            headers = self.raw_text[a:b].split("\n")
            for hit in headers:
                if not hit:
                    continue
                e,score,name = self._parse_spsa(hit)
                if e == None:
                    continue
                record.add_description(name,score,e)
                alignment = self._parse_alignment(name,self.raw_text[q:t])
                if not alignment:
                    alignment = blast_alignment()
                record.add_alignment(alignment)
            self.dataset.append(record)
            q = self.raw_text.find("Query= ",q+1)
        return self.dataset
            
    def _parse_spsa(self,hit):
        name = hit[:73]
        while name[-1] == " ":
            name = name[:-1]
        try:
            e = self._format_e(hit[73:])
            score = int(name[name.rfind(" ")+1:])
        except:
            return None,None,""
        name = name[:name.rfind(" ")].strip()
        if not name:
            name = "no name"
        return e,score,name
            
    def _parse_alignment(self,name,data):
        if len(name) > 20:
            name = name[:21]
        p = data.find(">"+name)
        if p == -1:
            return
        l = data.find("          Length = ",p)
        title = data[p+1:l]
        sbjct_length = int(data[l+19:data.find("\n",l)].replace(",",""))
        for symbol in ["\n","\r","\t","  "]:
            title = title.replace(symbol,"")
        hsps = []
        d = data.find(">",p+1)
        if d == -1:
            d = data.find("  Database:",p+1)
        block = data[p:d]
        b = block.find(" Score = ")
        while b > -1:
            q = block.find("Query:",b)
            score,e,aln_length,identities,positives,gaps,strand = self._parse_hsp_header(block[b:q])
            qlb,slb,qrb,srb,query,sbjct,hits = self._parse_hsp_body(block[q:block.find(" Score = ",q)])
            hsps.append(blast_hsp(score,e,aln_length,identities,positives,gaps,strand,qlb,slb,qrb,srb,query,sbjct,hits))
            b = block.find(" Score = ",b+1)
        alignment = blast_alignment(sbjct_length,title,score,e,hsps)
        return alignment
    
    def _parse_hsp_header(self,block):
        s = block.find(" Score = ")
        m = block.find(",   Method:")
        d = block.find(" Identities = ")
        p = block.find(" Positives = ")
        g = block.find("Gaps = ",d)
        t = block.find("Strand")
        score = float(block[s+9:block.find(" bits ")])
        if m > -1:
            e = self._format_e(block[block.find("Expect = ")+9:m])
        else:
            e = self._format_e(block[block.find("Expect = ")+9:block.find("\n",s)])
        aln_length = int(block[block.find("/",d)+1:block.find(" (",d)])
        identities = int(block[d+13:block.find("/",d)])
        positives = None
        if p > -1:
            positives = int(block[p+12:block.find("/",p)])
        gaps = 0
        if g > -1:
            gaps = int(block[g+7:block.find("/",g)])
        strand = ""
        if t > -1:
            strand = block[block.find("Plus",t):block.find("\n",t)]
            strand = strand.replace(" ","")
        return score,e,aln_length,identities,positives,gaps,strand
        
    def _parse_hsp_body(self,block):
        lines = block.split("\n")
        query = sbjct = hits = ""
        i = j = 0
        while i < len(lines):
            if i == 0:
                qlb = int(lines[i][6:lines[i].find(" ",8)])
                slb = int(lines[i+2][6:lines[i+2].find(" ",8)])
            if len(lines[i]) < 6 or lines[i][:6] != "Query:":
                i += 1
                continue
            indend = lines[i].rfind(" ",0,14)+1
            dedend = lines[i].find(" ",indend+1)
            query += lines[i][indend:dedend].upper()
            hits += lines[i+1][indend:dedend]
            sbjct += lines[i+2][indend:dedend].upper()
            j = i
            i += 3
        qrb = int(lines[j][lines[j].rfind(" "):])
        srb = int(lines[j+2][lines[j+2].rfind(" "):])
        return qlb,slb,qrb,srb,query,sbjct,hits
        
    def _format_e(self,e):  # e as string;
        if e.find(".") > -1:
            e = float(e)
        else:
            d = e.find("e-")
            try:
                x = int(e[:d])
            except:
                x = 1
            y = int(e[d+2:])
            e = x*10.0**(-y)
        return e

##############################################################################################################
class win_parser(sys_parser):
    def __init__(self,program,raw_text):
        sys_parser.__init__(self,program,raw_text)

##############################################################################################################
class linux_parser(sys_parser):
    def __init__(self,program,raw_text):
        sys_parser.__init__(self,program,raw_text)

    def _parse(self):
        self.raw_text = self.raw_text.replace("\r","")
        #### TEMP
        IO = IO()
        #IO.save(self.raw_text,"lintmp.out","text")
        if self.program in ("bl2seq","bl2seqp"):
            return self._parse_bl2seq()
        q = self.raw_text.find("Query= ")
        if q == -1:
            return
        while q != -1:
            t = self.raw_text.find("Query= ",q+1)
            l = self.raw_text.find("Length=",q+1)+7
            title = self.raw_text[q+7:self.raw_text.find("\n",q)]
            query_length = int(self.raw_text[l:self.raw_text.find("\n",l)].replace(",",""))
            record = blast_record(query_length,title)
            a = self.raw_text.find("Sequences producing significant alignments",q,t)
            if a == -1:
                q = self.raw_text.find("Query= ",q+1)
                continue
            else:
                a = self.raw_text.find("\n\n",a)+2
            b = self.raw_text.find(">",a)-2
            headers = self.raw_text[a:b].split("\n")
            for hit in headers:
                if not hit:
                    continue
                
                e,score,name = self._parse_spsa(hit)
                if e == None:
                    continue
                record.add_description(name,score,e)
                alignment = self._parse_alignment(name,self.raw_text[q:t])
                if not alignment:
                    alignment = blast_alignment()
                record.add_alignment(alignment)
            self.dataset.append(record)
            q = self.raw_text.find("Query= ",q+1)
        return self.dataset
            
    def _parse_bl2seq(self):
        q = self.raw_text.find("Query= ")
        l = self.raw_text.find("Length=",q+1)+7
        title = self.raw_text[q+7:self.raw_text.find("\n",q)]
        query_length = int(self.raw_text[l:self.raw_text.find("\n",l)].replace(",",""))
        record = blast_record(query_length,title)
        alignment = self._parse_bl2seq_alignment()
        record.add_description(alignment.title,alignment.get_score(),alignment.get_expect())
        record.add_alignment(alignment)
        self.dataset.append(record)
        return self.dataset
    
    def _parse_spsa(self,hit):
        name, score, e = [v for v in hit.strip().split(" ") if v]
        try:
            e = self._format_e(e)
            score = int(score)
        except:
            return None,None,""
        return e,score,name
            
    def _parse_alignment(self,name,data):
        if len(name) > 20:
            name = name[:21]
        p = data.find(">"+name)
        if p == -1:
            return
        l = data.find("Length=",p)+7
        title = data[p+2:data.find("\n",p)]
        sbjct_length = int(data[l:data.find("\n",l)].replace(",",""))
        for symbol in ["\n","\r","\t","  "]:
            title = title.replace(symbol,"")
        hsps = []
        d = data.find(">",p+1)
        if d == -1:
            d = data.find("  Database:",p+1)
        block = data[p:d]
        b = block.find(" Score = ")
        while b > -1:
            q = block.find("Query",b)
            score,e,aln_length,identities,positives,gaps,strand = self._parse_hsp_header(block[b:q])
            qlb,slb,qrb,srb,query,sbjct,hits = self._parse_hsp_body(block[q:block.find(" Score = ",b+1)])
            hsps.append(blast_hsp(score,e,aln_length,identities,positives,gaps,strand,qlb,slb,qrb,srb,query,sbjct,hits))
            b = block.find(" Score = ",b+1)
        alignment = blast_alignment(sbjct_length,title,score,e,hsps)
        return alignment
            
    def _parse_bl2seq_alignment(self):
        p = self.raw_text.find("Subject=")+8
        title = self.raw_text[p:self.raw_text.find("\n",p)]
        l = self.raw_text.find("Length=",p)+7
        sbjct_length = int(self.raw_text[l:self.raw_text.find("\n",l)].replace(",",""))
        for symbol in ["\n","\r","\t","  "]:
            title = title.replace(symbol,"")
        hsps = []
        block = self.raw_text[p:]
        b = block.find(" Score = ")
        while b > -1:
            q = block.find("Query",b)
            score,e,aln_length,identities,positives,gaps,strand = self._parse_hsp_header(block[b:q])
            b = block.find(" Score = ",b+1)
            qlb,slb,qrb,srb,query,sbjct,hits = self._parse_hsp_body(block[q:b])
            hsps.append(blast_hsp(score,e,aln_length,identities,positives,gaps,strand,qlb,slb,qrb,srb,query,sbjct,hits))
        alignment = blast_alignment(sbjct_length,title,score,e,hsps)
        return alignment
