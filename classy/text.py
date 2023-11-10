from .Struct import Struct
from numpy import array,log10

letters=[chr(x) for x in range(ord('a'),ord('z')+1)]

def letter_freq(language='English',feature_names=letters):

    ws="""Letter   	French    	German    	Spanish    	Esperanto    	Italian   	Turkish   	Swedish
    a 	7.636% 	6.51% 	12.53% 	12.12% 	11.74% 	11.68% 	9.3%
    b 	0.901% 	1.89% 	1.42% 	0.98% 	0.92% 	2.95% 	1.3%
    c 	3.260% 	3.06% 	4.68% 	0.78% 	4.5% 	0.97% 	1.3%
    d 	3.669% 	5.08% 	5.86% 	3.04% 	3.73% 	4.87% 	4.5%
    e 	14.715% 	17.40% 	13.68% 	8.99% 	11.79% 	9.01% 	9.9%
    f 	1.066% 	1.66% 	0.69% 	1.03% 	0.95% 	0.44% 	2.0%
    g 	0.866% 	3.01% 	1.01% 	1.17% 	1.64% 	1.34% 	3.3%
    h 	0.737% 	4.76% 	0.70% 	0.38% 	1.54% 	1.14% 	2.1%
    i 	7.529% 	7.55% 	6.25% 	10.01% 	11.28% 	8.27% 	5.1%
    j 	0.545% 	0.27% 	0.44% 	3.50% 	0.00% 	0.01% 	0.7%
    k 	0.049% 	1.21% 	0.00% 	4.16% 	0.00% 	4.71% 	3.2%
    l 	5.456% 	3.44% 	4.97% 	6.14% 	6.51% 	5.75% 	5.2%
    m 	2.968% 	2.53% 	3.15% 	2.99% 	2.51% 	3.74% 	3.5%
    n 	7.095% 	9.78% 	6.71% 	7.96% 	6.88% 	7.23% 	8.8%
    o 	5.378% 	2.51% 	8.68% 	8.78% 	9.83% 	2.45% 	4.1%
    p 	3.021% 	0.79% 	2.51% 	2.74% 	3.05% 	0.79% 	1.7%
    q 	1.362% 	0.02% 	0.88% 	0.00% 	0.51% 	0% 	0.007%
    r 	6.553% 	7.00% 	6.87% 	5.91% 	6.37% 	6.95% 	8.3%
    s 	7.948% 	7.27% 	7.98% 	6.09% 	4.98% 	2.95% 	6.3%
    t 	7.244% 	6.15% 	4.63% 	5.27% 	5.62% 	3.09% 	8.7%
    u 	6.311% 	4.35% 	3.93% 	3.18% 	3.01% 	3.43% 	1.8%
    v 	1.628% 	0.67% 	0.90% 	1.90% 	2.10% 	0.98% 	2.4%
    w 	0.114% 	1.89% 	0.02% 	0.00% 	0.00% 	0% 	0.03%
    x 	0.387% 	0.03% 	0.22% 	0.00% 	0.00% 	0% 	0.1%
    y 	0.308% 	0.04% 	0.90% 	0.00% 	0.00% 	3.37% 	0.6%
    z 	0.136% 	1.13% 	0.52% 	0.50% 	0.49% 	1.50% 	0.02%

    """


  
    if language=='English':
        wstr="""Letter   	Frequency  
        a 	8.167%
        b 	1.492%
        c 	2.782%
        d 	4.253%
        e 	12.702%
        f 	2.228%
        g 	2.015%
        h 	6.094%
        i 	6.966%
        j 	0.153%
        k 	0.772%
        l 	4.025%
        m 	2.406%
        n 	6.749%
        o 	7.507%
        p 	1.929%
        q 	0.095%
        r 	5.987%
        s 	6.327%
        t 	9.056%
        u 	2.758%
        v 	0.978%
        w 	2.360%
        x 	0.150%
        y 	1.974%
        z 	0.074%
        """

        lines=wstr.strip().split('\n')[1:]
        freq={}
        total=0.0
        for line in lines:
            letter,valstr=line.split()
            val=float(valstr[:-1])/100.0
            freq[letter]=val
            total+=val
            
        new_total=0.0
        for key in freq:
            freq[key]/=total
            new_total+=freq[key]
        total=new_total
        
        f_letter=Struct(freq)
        return [f_letter[_] for _ in feature_names]
        
        
    elif language in ws:
        
        lines=ws.strip().split('\n')
        languages=[x.strip() for x in lines[0].split()]
        i=languages.index(language)
        
        lines=lines[1:]
        
        freq={}
        total=0.0
        for line in lines:
            valstrs=line.split()
            letter=valstrs[0]
            val=float(valstrs[i][:-1])/100.0
            freq[letter]=val
            total+=val
            
        new_total=0.0
        for key in freq:
            freq[key]/=total
            new_total+=freq[key]
        total=new_total
        
        f_letter=Struct(freq)
        return [f_letter[_] for _ in feature_names]
        
        
    else:
        raise ValueError("Language %s not implemented" % language)
     
from sklearn.feature_extraction.text import CountVectorizer  
   
def count_letters(filenames):
    from glob import glob
    # with spaces
    # CountVectorizer(token_pattern=r'[A-Za-z ]',min_df=1)
    #Just letters, no spaces
    filenames=glob(filenames)
    
    text=[]
    vectorizer=CountVectorizer(token_pattern=r'[A-Za-z]',min_df=1)
    analyze = vectorizer.build_analyzer()    
    for filename in filenames:
        with open(filename) as fid:
            mytext=fid.read()
            #mytext=mytext.decode('utf8','ignore')            
            text.append(mytext)
        
    X=vectorizer.fit_transform(text).toarray()
    return X,[str(_) for _ in vectorizer.get_feature_names_out()]     
     
class LanguageFileClassifier(object):
    
    def __init__(self):
        self.target_names=['English','French','German','Spanish',
                           'Esperanto','Italian','Turkish','Swedish']
        self.feature_probability=array( [letter_freq(lang) for lang in self.target_names] )
        self.feature_names=letters
        
    def loglikelihood(self,filenames,verbose=False):
        from glob import glob
        import warnings
    
        # with spaces
        # CountVectorizer(token_pattern=r'[A-Za-z ]',min_df=1)
        #Just letters, no spaces
        filenames=glob(filenames)
        results=[]
        for filename in filenames:
            if verbose:
                print(filename)
                
            count,feature_names=count_letters(filename)
            result=[]
            for lang in self.target_names:
                p=letter_freq(lang,feature_names)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    value=(log10(p)*count).sum()
                result.append(value)
                if verbose:
                    print("\t%s: %f" % (lang,value))
                
            results.append(result)

            
            
        return array(results)
        
        
    def predict(self,filenames,verbose=False):
        L=self.loglikelihood(filenames,verbose)
        return L.argmax(axis=1)
        
    def predict_names(self,filenames,verbose=False):
        result=self.predict(filenames,verbose)
        
        return [self.target_names[i] for i in result]
        
def load_files(dirname,verbose=False):
    import sklearn.datasets
    
    data=sklearn.datasets.load_files(dirname,encoding='utf8',decode_error='ignore',shuffle=False)
    data.targets=data.target
    
    if verbose:
        print(len(data.filenames)," filenames")
        print(data.target_names, " target names")
    
    return data

def text_to_vectors(dirname_or_textdata,test_dirname_or_textdata=None,ngram_range=(1, 1),verbose=False):
    if isinstance(dirname_or_textdata,str):
        textdata=load_files(dirname_or_textdata,verbose)
    else:
        textdata=dirname_or_textdata

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    vectors = vectorizer.fit_transform(textdata.data)
    
    data=Struct()
    data.vectorizer=vectorizer
    data.vectors=vectors
    data.targets=textdata.targets
    data.target_names=textdata.target_names
    data.feature_names=vectorizer.get_feature_names_out()
    
    if not test_dirname_or_textdata is None:
        if isinstance(test_dirname_or_textdata,str):
            textdata=load_files(test_dirname_or_textdata,verbose)
        else:
            textdata=test_dirname_or_textdata

        test_vectors = vectorizer.transform(textdata.data)
        test_data=Struct()
        test_data.vectorizer=vectorizer
        test_data.vectors=test_vectors
        test_data.targets=textdata.targets
        test_data.target_names=textdata.target_names
        test_data.feature_names=vectorizer.get_feature_names_out()
        
        return data,test_data
    else:
        return data
    

from sklearn.naive_bayes import MultinomialNB

class Multinomial(MultinomialNB):
    def percent_correct(self,vectors,targets):
        return self.score(vectors,targets)*100.0

    def predict_names(self,vectors):
        result=self.predict(vectors)
        
        return [self.target_names[i] for i in result]


    def __init__(self,**kwargs):
        if 'fit_prior' not in kwargs:
            kwargs['fit_prior']=False
            
        MultinomialNB.__init__(self,**kwargs)

    
