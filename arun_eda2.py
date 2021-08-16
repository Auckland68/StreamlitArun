import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import sklearn
from wordcloud import WordCloud, STOPWORDS
from spellchecker import SpellChecker
from textblob import TextBlob
import matplotlib.pyplot as plt
import pickle
import string
import joblib
import SessionState
from collections import Counter
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Flatten,Embedding,Dropout
from keras.models import model_from_json
#st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Arun District Travel Review Data")
st.sidebar.subheader("Dashboards")
dashboard_choice = st.sidebar.selectbox("Please Choose Dashboard",("Exploratory Data Analysis","Keyword Analysis","Review Analyser"),key = "main")
st.markdown("This application is a Streamlit Dashboard to analyse Tourist Reviews in Arun District️")
st.markdown("Please note it is a demonstration application only and is based on a sample of reviews from the dataset")
#st.sidebar.title("Arun District Travel Review Data 2019")
st.sidebar.subheader("Arun District Travel Review Data")

# Load dataset and cache the output
DATA_URL = ("new_data.csv")

@st.cache(allow_output_mutation=True, max_entries = 10, ttl=3600)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

def open_tok(name):
    with open(name, 'rb') as handle:
        file = pickle.load(handle)
        return file

# load json and create model
json_file = open('model10.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model10.h5")
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load tokenizers and models
accom_tok = open_tok("accom_tokprot4.pickle")
food_tok = open_tok("food_tokprot4.pickle")
attract_tok = open_tok("attract_tokprot4.pickle")
sent_tok = open_tok('sent_tokprot4.pickle')

model1 = joblib.load('best_model_accom.sav')
model2 = load_model('best_model_food.h5')
model3 = loaded_model
model4 = open_tok('model4prot4.pickle')


# Functions
if dashboard_choice == "Exploratory Data Analysis":
    if st.sidebar.checkbox('Show Raw Data'):
        st.subheader('Arun District Review Data')
        st.text("Reviews are classed as postive relating to user review ratings of 4 and 5, neutral - ratings of 3, and negative - ratings of 1 and 2. For more information on the features please refer to the main report.")
        st.write(data)
    # Location of Users and time of review on a map
    st.sidebar.subheader("Visitor Locations")
    st.sidebar.markdown("Location of Visitors posting reviews for each category")
    select_users = st.sidebar.selectbox('Visualisation Type',['Map','Pie chart'], key = "loc")
    cat = st.sidebar.selectbox("Choose Category of Review",("Accommodation","Food","Attractions"),key = "cat_Loc")
    modified_data = data[data.category == cat]
    review_locs = modified_data["location"].value_counts()
    review_locs = pd.DataFrame({"Visitor Location":review_locs.index,"Number of Reviews":review_locs.values})
    if select_users == "Map":
        st.subheader("Map of Visitor Location")
        st.markdown("Zoom in for detail")
        st.markdown(cat)
        st.map(modified_data)
    else:
        fig = px.pie(review_locs, values = "Number of Reviews", names = "Visitor Location")
        st.subheader("Visitor Location")
        st.markdown(cat)
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Show Monthly Data",True, key = "r"):
       st.subheader("Reviews posted by Month")
       test_series = pd.DataFrame(data["post month"].value_counts().reset_index())
       test_series.columns = ["Month","Reviews"]
       fig = px.bar(test_series, x = "Month", y = "Reviews")
       st.plotly_chart(fig)

    # Overall sentiment for Arun District By category
    st.sidebar.subheader("User Sentiment by Category for Arun District")
    select_chart = st.sidebar.selectbox('Visualisation Type',['Histogram','Pie chart'], key = "sel_char")
    cat_sentiment = st.sidebar.selectbox("Choose Category of Review",("Accommodation","Food","Attractions"),key = "cat_sent")
    sent_count = data[data["category"] == cat_sentiment]["sentiment"].value_counts()
    sent_count = pd.DataFrame({"Sentiment":sent_count.index,"Number of Reviews":sent_count.values})
    if st.sidebar.checkbox("Show",True, key = "s"):
        if select_chart == "Histogram":
            st.subheader("Sentiment By Category for Arun District")
            st.subheader("%s" % (cat_sentiment))
            fig = px.bar(sent_count, x = "Sentiment", y = "Number of Reviews", color = "Number of Reviews", height = 500)
            st.plotly_chart(fig)
        else:
            fig = px.pie(sent_count, values = "Number of Reviews", names = "Sentiment")
            st.subheader("Sentiment By Category for Arun District")
            st.subheader("%s" % (cat_sentiment))
            st.plotly_chart(fig)


    # Most highly reviewed establishments
    st.header("Most Reviewed Establishments By Town, Category and Sentiment")
    st.sidebar.subheader("Most Reviewed Establishments By Town,Category and Sentiment")
    cat_choices = st.sidebar.selectbox("Choose Category", ("Accommodation","Food","Attractions"),key = "cats")
    town_choices = st.sidebar.selectbox("Choose Town", ("Arundel","Bognor","Littlehampton"), key = "towns")
    sentiment = st.sidebar.selectbox("Choose Sentiment",("positive","negative","neutral"),key = "sent_choice")
    name_counts =data[(data["town"] == town_choices) & (data["category"] == cat_choices) &(data["sentiment"]==sentiment)]["name"].value_counts()
    name_counts = pd.DataFrame({"Name":name_counts.index,"Number of Reviews":name_counts.values})
    if st.sidebar.checkbox("Show",True, key = "m"):
        st.subheader("Highest Number of Reviews By Town,Category,Establishment and Sentiment")
        st.subheader("%s %s %s" % (town_choices,cat_choices,sentiment))
        fig = px.bar(name_counts, x="Name", y="Number of Reviews",color = "Number of Reviews", width = 800, height = 500)
        st.plotly_chart(fig)


    #  Detail Analysis Sentiment By Town, Category and estblishment type
    st.sidebar.header("Detail Analysis: Sentiment By Category, Town and Establishment Type")
    if st.sidebar.checkbox("Show", True, key = "tce"):
        select = st.sidebar.selectbox('Visualisation Type',['Histogram','Pie chart'], key = "sel")
        town_choice = st.sidebar.selectbox("Choose a Town",("Arundel","Bognor","Littlehampton"), key = "tc")
        category_choice = st.sidebar.selectbox("Please Choose a Category",('Accommodation','Food','Attractions'),key = "cc")
        if category_choice == "Accommodation":
            type_choice = st.sidebar.multiselect("Please Choose at least one Establishment Type",['Hotel', 'B&B/Inn', 'AccomOther'])
        elif category_choice == "Food":
            type_choice = st.sidebar.multiselect("Please Choose at least one Establishment Type",['Restaurant', 'Pub/Bar', 'Café', 'Steakhouse/Diner',
               'Fast Food/Takeaway', 'Gastropub'])
        else:
            type_choice = st.sidebar.multiselect("Choose at least one Establishment Type",['Historical/Culture', 'Nature/Gardens', 'Amusements/Fun', 'Nightlife',
               'Beach/Outdoor', 'Shopping', 'Spas/Leisure Centres','Classes/Workshops'])

        st.header('Sentiment By Category, Town and Establishment Type')
        st.markdown("Please enter at least one establishment type from the menu on the sidebar")
        st.subheader("%s %s %s" % (town_choice, category_choice, type_choice))
        sentiment_counts = data[(data["town"] == town_choice) & (data["category"] == category_choice) & (data["type"].isin(type_choice))]["sentiment"].value_counts()
        sentiment_counts = pd.DataFrame({"Sentiment":sentiment_counts.index,"Number of Reviews":sentiment_counts.values})
        if select == "Histogram":
            fig = px.bar(sentiment_counts, x = "Sentiment", y = "Number of Reviews", color = "Number of Reviews", height = 500)
            st.plotly_chart(fig)
        else:
            fig = px.pie(sentiment_counts, values = "Number of Reviews", names = "Sentiment")
            st.plotly_chart(fig)


    # WordClouds for Postive and Negative sentiment_count
    st.sidebar.header("Word Cloud")
    fig, ax = plt.subplots()
    if st.sidebar.checkbox("Show", True, key='7'):
        town_choice2 = st.sidebar.selectbox("Choose a Town",("Arundel","Bognor","Littlehampton"), key = "WC")
        category_choice2 = st.sidebar.selectbox("Choose a Category",('Accommodation','Food','Attractions'))
        word_sentiment = st.sidebar.radio('Choose sentiment for Word Cloud', ('positive', 'neutral', 'negative'))
        word_cloud = data[(data['category']== category_choice2) & (data["town"] == town_choice2) & (data['sentiment'] == word_sentiment)]
        st.subheader('Word cloud for %s sentiment' % (word_sentiment))
        st.subheader("%s %s" % (town_choice2, category_choice2))
        words = " ".join(title for title in word_cloud.title)
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        st.pyplot(fig)

# Allow random review to be shown
    st.sidebar.subheader("Show Random Reviews for Chosen Categories and Town")
    random = word_cloud[["title","review","post month"]].sample(1)
    st.sidebar.markdown(random.iat[0,0])
    if st.sidebar.checkbox("Show Full Review", True, key = "8"):
        st.sidebar.markdown(random.iat[0,1])
        st.sidebar.markdown("Posted in Month:")
        st.sidebar.markdown(random.iat[0,2])

elif dashboard_choice == "Keyword Analysis":

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

    # Get keywords by town, category and sentiment
    st.sidebar.subheader("Keywords By Town, Category and Sentiment")
    town_pick = st.sidebar.selectbox("Choose Town", ("Arundel","Bognor","Littlehampton"), key = "keytown")
    cat_pick = st.sidebar.selectbox("Choose Category", ("Accommodation","Food","Attractions"),key = "keycat")
    sentiment_pick = st.sidebar.selectbox("Choose Sentiment",("positive","negative","neutral"),key = "keysent")
    word_pick = st.sidebar.selectbox("Choose Analysis Type",("words","bigrams","noun phrases"),key = "w_choice")
    chosen = data[(data["town"] == town_pick) & (data["category"] == cat_pick) &(data["sentiment"]==sentiment_pick)]

    def get_nouns(df):
        df["review_lower"] = df["review"].apply(lambda x: x.strip().lower())
        df["review_lower"] = df["review_lower"].str.replace(r'\bread less$', '', regex=True).str.strip()
        df["review_lower"] = df["review_lower"].apply(lambda x: contractions.fix(x))
        df["token"] = df["review_lower"].apply(lambda x: nltk.word_tokenize(x))
        punc = string.punctuation
        df["review_punc"] = df["token"].apply(lambda x: [word for word in x if word not in punc])
        df["review_asc"] = df["review_punc"].apply(lambda x: [e for e in x if e.encode("ascii","ignore")])
        stop = stopwords.words('english')
        df["stop"] = df["review_asc"].apply(lambda x: [w for w in x if w not in stop])
        df['pos_tags'] = df['stop'].apply(nltk.tag.pos_tag)
        df['nouns'] = df['pos_tags'].apply(lambda x: [i[0] for i in x if i[1].startswith('NN')])
        df["nouns_adj"] = df['pos_tags'].apply(lambda x: [i[0] for i in x if i[1].startswith('NN') or i[1].startswith('JJ')])
        df['bigrams'] = df['nouns'].apply(lambda x: ngrams(x,2))
        df["all_nouns"] = df.nouns.apply(lambda x: Counter(x))
        df["all_bigrams"] = df.bigrams.apply(lambda x: Counter(x))
        df["all_noun_adj"] = df.nouns_adj.apply(lambda x: ngrams(x,2))
        df["all_noun_adj"] = df.all_noun_adj.apply(lambda x: Counter(x))

        return df

    key_df = get_nouns(chosen)

    def count_total(df,col):
        df = df[col].sum()
        df_top = df.most_common(15)
        return df_top

    key_words = pd.DataFrame(count_total(key_df,"all_nouns"),columns =["Word","Freq"])
    key_bigrams = pd.DataFrame(count_total(key_df,"all_bigrams"),columns = ["Word","Freq"])
    key_bigrams["Word"] = key_bigrams["Word"].apply(lambda x: ' '.join(x))
    key_noun_adj = pd.DataFrame(count_total(key_df,"all_noun_adj"),columns = ["Word","Freq"])
    key_noun_adj["Word"] = key_noun_adj["Word"].apply(lambda x: ' '.join(x))


    if st.sidebar.checkbox("Show",True,key = "n"):
        st.subheader("Keywords By Town, Category and Sentiment")
        st.subheader("%s %s %s %s" % (town_pick,cat_pick,sentiment_pick,word_pick))
        st.text("Hover over the chart for numbers")
        if word_pick == "words":
            fig = px.bar(key_words, x = "Word",y = "Freq")
        elif word_pick =="bigrams":
            fig = px.bar(key_bigrams, x = "Word",y = "Freq")
        else:
            fig = px.bar(key_noun_adj, x = "Word",y = "Freq")
        st.plotly_chart(fig)

    st.markdown("Note: Single words provide overall aspects of interest, pairs of nouns or bigrams can be more informative, as can choosing noun phrases which usually pair a noun with an adjective")

# Sentiment Analyser
else:
    # Function to clean sentences
    def process(text):
        if text != []:
                text = text.replace('\n',' ')
                text = text.strip().lower()
                text = text.replace('xmas','christmas')
                text = text.replace('\£',"")
                text = text.replace(r'\/'," ")
                text = text.replace('\d+\-\d+',"")
                text = text.replace('\d+\w{2}',"")
                text = text.replace('\.{3,}',"")
                text = text.replace(' i ',"")
                text = text.replace(' le ',"")
                text = contractions.fix(text)
                text = nltk.word_tokenize(text)
                punc = string.punctuation
                text = [word for word in text if word not in punc]
                text = [n for n in text if not n.isnumeric()]
                text = [e for e in text if e.encode("ascii","ignore")]
                stop = stopwords.words("english")
                stop_remove = ["not","don't","didn't","wasn't","won't","isn't"]
                stop1 = [w for w in stop if w not in stop_remove]
                add_stop = ['etc','read','read less','lot','butlins', 'bognor','regis','b',' i ','..','arundel castle','premier','inn','u',
                            'castle',"year","hilton","time","day","shoreline","oyster","bay","church farm","hotham","hotham park",
                            "hawk walk","hawk","arundel","littlehampton"]
                stop1.extend(add_stop)
                text = [w for w in text if w not in stop1]
                lemmatizer = WordNetLemmatizer()
                text = [lemmatizer.lemmatize(w) for w in text ]
                spell = SpellChecker()
                word_list = []
                for w in text:
                    new = spell.correction(w)
                    if new != w:
                        word_list.append(new)
                    else:
                        word_list.append(w)
                    text = ' '.join(word_list)


        return text

    ## - Aspect Extraction
    def extract(text):
        text = word_tokenize(text)
        text_pos = nltk.tag.pos_tag(text)
        noun = [i[0] for i in text_pos if i[1].startswith('N')]
        return noun

    ## - Construct Dataframe
    def construct(review_text):
        sentences = pd.DataFrame(nltk.sent_tokenize(review_text),columns = ["Sentences"])
        sentences["cleaned"] = sentences["Sentences"].apply(lambda x: process(x))
        sentences = sentences[sentences.astype(str)['cleaned'] != '[]']
        sentences["extract_noun_phrases"] = sentences["cleaned"].apply(phrase_extract)
        sentences['extract_noun_phrases'] = np.where(sentences['extract_noun_phrases'].str.len() == 0,
                                                     sentences['cleaned'], sentences['extract_noun_phrases'])
        sentences["joined_phrases"] = sentences["extract_noun_phrases"].apply(lambda x: ' '.join(x) if type(x) != str else x)
        sentences["extract_nouns"] = sentences["cleaned"].apply(extract)
        sentences["joined_nouns"] = sentences["extract_nouns"].apply(lambda x: ' '.join(x))
        sentences["joined_nouns"] = sentences["joined_nouns"].apply(lambda x: "general" if x == "" else x)
        return sentences

    ## Phrase Extraction & Encoding Text
    def phrase_extract(text):
        blob = TextBlob(text)
        return blob.noun_phrases

    def encode(nouns,tokenizer):
        x_s = tokenizer.texts_to_sequences(noun)
        x_w = pad_sequences(np.array(x_s, dtype = "object"), maxlen = 100,padding = "post", truncating = "post", value = 0.0)
        return x_w

    def encode2(text,vectorizer):
        enc = vectorizer.transform(text)
        return enc

    ## Predict
    def predict1(model,X):
        y_pred = model.predict(X)
        return y_pred

    ## sentiment
    def sent(df):
        label_sent = {1:"Negative",0:"Positive"}
        encoded_sent = encode2(df["joined_phrases"],sent_tok)
        sent_predict = pd.DataFrame(predict1(model4, encoded_sent),columns = ["S"])
        sent_probs = model4.predict_proba(encoded_sent)[:,1]
        sent_predict["Predicted Sentiment"] = sent_predict["S"].apply(lambda x: label_sent.get(x))
        sentiment_summary = pd.concat([df["Sentences"],sent_predict],axis = 1)
        return sentiment_summary

    ## Scoring
    def scoring(x):
        if x == "Positive":
            score = 1
        else:
            score = -1
        return score

    ## Aspect Prediction
    def aspect_all(df):
        pos_df = df[df["Predicted Sentiment"] == "Positive"]
        neg_df = df[df["Predicted Sentiment"] == "Negative"]
        pos_df = pd.DataFrame(pos_df["Predicted Aspect"].value_counts())
        neg_df = pd.DataFrame(neg_df["Predicted Aspect"].value_counts())
        pos_df.columns = ["Num Pos"]
        neg_df.columns = ["Num Neg"]
        df_sent = pd.concat([pos_df,neg_df],axis = 1)
        df_sent = df_sent.fillna(0)
        df_sent["Total"] = df_sent["Num Pos"]+df_sent["Num Neg"]
        df_sent["%Pos"] = round(df_sent["Num Pos"]/(df_sent["Num Pos"] + df_sent["Num Neg"]),2)*100
        df_sent["%Neg"] = round(df_sent["Num Neg"]/(df_sent["Num Pos"] + df_sent["Num Neg"]),2)*100
        df_sent.sort_index(inplace = True)
        return df_sent

    ## chart
    def graph_sentiment_numbers(df):

        labels = list(df.index)
        pos = list(df["%Pos"])
        neg = list(df["%Neg"]*-1)

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize = (8,6))
        ax.bar(x - width/2, pos, width, label='Pos',color = "lightsteelblue")
        ax.bar(x + width/2, neg, width, label='Neg',color = "slategrey")

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_title("Review - Positive and Negative Aspects",pad = 20, fontsize = 15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("%")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(frameon = False, loc = "best")
        fig.tight_layout()
        st.pyplot(fig)


    ## Review Analyser
    def review_analyser(review,category):

    # Labels
        label_accom = {0:'Entertainment',1:'Food',2:'Hotel',3:'Location',4:'Room',5:'Staff',6:'Value'}
        label_food = {0:"Food Quality",1:"Meal Exp",2:"Menu Choice",3:"Staff",4:"Value",5:"Visit Exp"}
        label_attract = {0:"Activities",1:"Amenities",2:"History",3:"Nature",4:"Staff/Service",5:"Value",6:"Visit Exp"}
        label_sent = {0:"Positive",1:"Negative"}


        # Models with weights
        accom = model1
        food = model2
        attract = model3
        sentiment = model4

        # Split review into sentences, extract noun_phrases and nouns and clean. Where extracted noun_phrases is an empty list
        # the cleaned column is inserted instead.
        sentences = construct(review)

        # Predict aspects (Deep Learning Model)
        if category == "Food":
            s = food_tok.texts_to_sequences(sentences["joined_nouns"])
            w = pad_sequences(np.array(s, dtype = "object"), maxlen = 20,padding = "post", truncating = "post", value = 0.0)
            predict = predict1(model2,w)
            y_pred_class = np.argmax(model2.predict(w), axis=-1)
            y_pred_class = pd.DataFrame(y_pred_class, columns = ["A"])
            y_pred_class["Predicted Aspect"] = y_pred_class["A"].apply(lambda x: label_food.get(x))

            # Predict sentiment (TFIDF)
            sentiment_summary = sent(sentences)

        # Predict aspects (TFIDF SVM Model)
        elif category == "Accommodation":
            encoded = accom_tok.transform(sentences["joined_nouns"])
            predict = predict1(model1,encoded)
            y_pred_class = pd.DataFrame(predict, columns = ["A"])
            y_pred_class["Predicted Aspect"] = y_pred_class["A"].apply(lambda x: label_accom.get(x))

            # Predict sentiment (TFIDF)
            sentiment_summary = sent(sentences)

        else: # attractions
            s = attract_tok.texts_to_sequences(sentences["joined_nouns"])
            w = pad_sequences(np.array(s, dtype = "object"), maxlen = 100,padding = "post", truncating = "post", value = 0.0)
            predict = predict1(model3,w)
            y_pred_class = np.argmax(model3.predict(w), axis=-1)
            y_pred_class = pd.DataFrame(y_pred_class, columns = ["A"])
            y_pred_class["Predicted Aspect"] = y_pred_class["A"].apply(lambda x: label_attract.get(x))

            # Predict sentiment (TFIDF)
            sentiment_summary = sent(sentences)

        aspect_sentiment = pd.concat([y_pred_class,sentiment_summary],axis = 1)
        aspect_sentiment["Score"] = aspect_sentiment["Predicted Sentiment"].apply(lambda x: scoring(x))
        pos_neg = aspect_sentiment[aspect_sentiment["Predicted Sentiment"] != "Neutral"]
        sent_percent = aspect_all(pos_neg)
        graph_sentiment_numbers(sent_percent)

        return aspect_sentiment

    st.header("Review Sentiment Analyzer Tool")
    st.subheader("Please enter the text you'd like to analyse.")
    st.sidebar.markdown("Please enter the category you would like to analyse")
    st.markdown("Note: reviews should be at least one sentence. Please select an analysis category from the sidebar")

    state = SessionState.get(key = 0)
    ta_placeholder = st.empty()
    if st.button('Clear'):
        state.key +=1

    category = st.sidebar.selectbox('Category Type',['Accommodation','Food','Attractions'], key = "ct")
    review = ta_placeholder.text_area('Enter review text',value = '',height = None,max_chars = 5000)

    if st.button("Analyse"):
        sent_nums = sent_tokenize(review)
        if len(sent_nums) >= 1:
            with st.spinner("Analysing the text"):
                df = review_analyser(review,category)
                df.drop(columns = ["A","S","Score","Sentences"],axis = 1, inplace = True)
                st.write(df)
