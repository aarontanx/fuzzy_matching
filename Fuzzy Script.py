import pandas as pd
import os
from tqdm import tqdm
import re
import numpy as np
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

output_dir = r'output_location'

## Fuzzy matching threshold
fuzzyMatchingThreshold = 0.94

## Chunksize for matching
chunksizeBlkLst = 1000

## Fuzzy Algo
## Return the levenshtein edit distance between two strings *a* and *b*

def levenshtein_ratio(a,b):
    
    # Change string to uppercase
    a = a.upper().strip()
    b = b.upper().strip()
    
    # Return levenshtein edit distance between 2 strings
    if a == b:
        return 1
    if len(a) < len(b):
        a, b = b, a
    if not a:
        return len(b)
    previous_row = range(len(b) + 1)
    for i, column1 in enumerate(a):
        current_row = [i+1]
        for j, column2 in enumerate(b):
            insertions = previous_row[j+1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (column1 != column2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        ratio = (len(a) + len(b) - previous_row[-1]) / (len(a) + len(b))
    return ratio
    
## Function for n-grams
def ngrams(string, n=3):
    string = string.encode("ascii", errors = "ignore").decode()
    string = string.lower()
    chars_to_remove = [')', '(', '.', '|', '[', ']', '{', '}', "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string) # remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ').replace('-', ' ')
    string = string.title() # Capital at start of each word
    string = re.sub(' +', ' ',string).strip()
    string = ' '+ string + ' ' # pad
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]
    
# function for finding best match string            
def tfidf_match(list1, list2):
    
    # Firstly use ngram to vectorize the word
    # Then using those vectorized ngram, find their closest neighbor
    
    """
    Example:
    list1 = ['Hello', 'Hey']
    list2 = ['Hey', 'yEllo']
    
    First we fit and transform the vectorizer model using the first list, and then using the same model we transform it with second list.
    list1 will return 
      (0, 6)	0.47107781233161794
      (0, 5)	0.47107781233161794
      (0, 3)	0.47107781233161794
      (0, 1)	0.47107781233161794
      (0, 0)	0.33517574332792605
      (1, 4)	0.6316672017376245
      (1, 2)	0.6316672017376245
      (1, 0)	0.4494364165239821
    
    list2 will return the following matrix
      (0, 4)	0.6316672017376245
      (0, 2)	0.6316672017376245
      (0, 0)	0.4494364165239821
      (1, 6)	0.5773502691896257
      (1, 5)	0.5773502691896257
      (1, 3)	0.5773502691896257
    
    This is what available in our feature
    vectorizer.get_feature_names()
    [' He', 'Hel', 'Hey', 'ell', 'ey ', 'llo', 'lo ']
    Note that yEllo only have 3 matrix score, because only 'ell', 'llo', 'lo ' exists in the feature list.
    
    With all these in place, we then use nearest neighbor to comapre each sparse matrix list1[0] with list2[0] and list2[1], and list1[1] with list2[0] and list2[1].
    Applying this we will not need to compare word by word with loop, but a vectorized version of texts.
    
    We pick the closest one and then apply levenshtein score to create another scoring for filtering.
    """
    vectorizer = TfidfVectorizer(analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(list2)
    nbrs = NearestNeighbors(n_neighbors= 1, n_jobs= 1).fit(tfidf)
    distances, indices = nbrs.kneighbors(vectorizer.transform(list1))
    
    matches = [(round(distances[i][0], 2), list1[i], list2[j[0]]) for i, j in enumerate(indices)]
    matches = pd.DataFrame(matches, columns=['score', 'original', 'matched'])
    return matches
            
            
 # Data Cleansing
 def clean_data(df, column):       
     rx = r'[A-Za-z0-9]+'
     df['unique_id'] = df.index
     df[[column]] = df[[column]].fillna('')
     cleanColName = column+'_Clean'
     df[[cleanColName]] = df[[column]].apply(lambda x: re.sub(rx, '', str(x)).upper().strip())
     return df, cleanColName

            
       
def fuzzyMatching(leftTableNameMatching, rightTableNameMatching, fuzzyMatchingThreshold, chunksizeBlkLst):
    print('Starting Fuzzy Logic')
    
    for i in range(len(leftTableNameMatching['leftTableName'])):
        leftTableName = leftTableNameMatching['leftTableName'][i]
        
        # Table Column add suffix
        leftTable = leftTableNameMatching['leftTableData'][i].add_suffix('_Left_Table').copy()
        
        # Create unique ID for join later
        leftTable['leftTableId'] = range(len(leftTable))
        
        # Name Field add same suffix
        leftTableNameFieldList = [ i + '_Left_Table' for i in leftTableNameMatching['leftTableNameField'][i]]
        
        leftDf = leftTable.copy() # Keep one set for joining later
        leftTable = leftTable[leftTableNameFieldList].drop_duplicates() # Remove Duplicates
        
        # Convert columns into one list
        t = pd.DataFrame()
        for x in leftTableNameFieldList:
            t = t.append(leftTable[[x]].rename(columns = {x: 'leftFieldName'}), ignore_index = True)
        leftTable = t['leftFieldName'].drop_duplicates().tolist()
        del t
        
        for j in range(len(rightTableNameMatching['rightTableName'])):
            rightTableName = rightTableNameMatching['rightTableName'][j]
            rightTable = rightTableNameMatching['rightTableData'][j].add_suffix('_right_Table').copy()
            rightTableNameFieldList = [ j + '_right_Table' for j in rightTableNameMatching['rightTableNameField'][j]]
            
            rightDf = rightTable.copy() # Keep one set for joining later
            rightTable = rightTable[rightTableNameFieldList].drop_duplicates() # Remove Duplicates
            
            # Convert columns into list
            t = pd.DataFrame()
            for x in rightTableNameFieldList:
                t = t.append(rightTable[[x]].rename(columns = { x: 'rightFieldName'}), ignore_index = True)
            rightTable = t['rightFieldName'].drop_duplicates().tolist()
            del t
            
            # Fuzzy Logic Start here
   
            n = chunksizeBlkLst
            result = pd.DataFrame()
            
            for z in tqdm(range(int(round(len(rightTable)/n + 0.5, 0)))):
                print('Blacklist chunk count = ', z + 1)
                t = tfidf_match(leftTable, rightTable[ z*n: min((z+1)*n, len(rightTable)) ])
                t['fuzzy matching ratio'] = t.apply(lambda x: levenshtein_ratio(x['original'], x['matched']), axis = 1)
                t = t.loc(t['fuzzy matching ratio'] >= fuzzyMatchingThreshold]
                
                t = t[['original', 'fuzzy matching ratio', 'matched']]
                
                result = result.append(t, ignore_index= True)
                del t
                
            # Merging back to leftDf
            t = pd.DataFrame()
            for x in leftTableNameFieldList:
                t = t.append(leftDf[[x, 'leftTableId']], left_on = 'original', right_on = x, how = 'inner'), ignore_index = True)
                t = t[['leftTableId', 'fuzzy matching ratio', 'original', 'matched']].merge(leftDf, on='leftTableId', how='inner')
            result = t.copy()
            result = result.rename(columns={'matched': 'PO name'})
            
            # Merging back to rightDf
            t = pd.DataFrame()
            for i in rightTableNameFieldList:
                t = t.append(result.merge(rightDf, left_on = 'PO name', right_on = i, how = 'inner'), ignore_index = True)
            result = t.copy()
            del t
            
            # Table cleansing
            result = result.drop_duplicates()
            result.sort_values(by=['leftTableId', 'fuzzy matching ratio'], ascending = False)
            
            result = result[result['original'] != '']
            
            # Write to csv for each right table completion
            fileName = '\\Fuzzy_Matched_' + str(leftTableName) + '_' + str(rightTableName) + '.csv'
            
            output_path = output_dir + fileName
            
            result.to_csv(output_path, header = True, index = False)
    print('End of Fuzzy Logic')
    
 

tablename, colname = clean(tablename)
            
leftTableNameMatching = {
    'leftTableName' : ['tablename'], # Can have multiple item in this list
    'leftTableData' : [tablename], # Can have multiple item in this list
    'leftTableNameField': [[colname]] # Can have multiple item in this list
}

rightTableNameMatching = {
    'rightTableName' : ['tablename'], # Can have multiple item in this list
    'rightTableData' : [tablename], # Can have multiple item in this list
    'rightTableNameField': [[colname]] # Can have multiple item in this list
}

fuzzyMatching(leftTableNameMatching, rightTableNameMatching, fuzzyMatchingThreshold, chunksizeBlkLst)
