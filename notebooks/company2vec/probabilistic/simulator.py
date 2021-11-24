from scipy.sparse import csr_matrix
from scipy.stats import beta, binom, dirichlet, multinomial
import numpy as np

def generate_topics(industries, words_p_topic, concentration, background_words):
    word2id = {}
    id2word = {}
    ind = 0
    for industry_name in industries:
        for i in range(words_p_topic):
            word = f"{industry_name}_word_{i}"
            word2id[word] = ind
            id2word[ind] = word
            ind += 1

    for i in range(background_words):
        word=f"background_word_{i}"
        word2id[word] = ind
        id2word[ind] = word
        ind += 1

    topics = np.zeros(shape=(len(industries), len(word2id)))
    alpha_background = np.ones(shape=(len(word2id),))
    alpha_background[(-background_words):] = concentration
    background_topic = dirichlet.rvs(alpha_background)[0]

    for i,_ in enumerate(industries):
        alpha = np.ones(shape=(len(word2id,)))
        alpha[(i*words_p_topic):((i+1)*words_p_topic)] = concentration
        topics[i,:] = dirichlet.rvs(alpha)

    return topics, background_topic, word2id, id2word

def generate_companies(industries, 
                       num_companies=1000, 
                       words_p_company=100, 
                       words_p_topic=5, 
                       concentration=10,
                       background_words=100,
                       a=1.0,
                       b=1.0,
                      ):
    num_industries = len(industries)
    topics, background_topic, word2id, id2word = generate_topics(industries, words_p_topic, concentration, background_words)
    n_topics, n_words = topics.shape
    companies = np.zeros(shape=(num_companies, n_words))
    Z = []
    company_industry = []
    for i in range(num_companies):
        industry = np.random.choice(num_industries)
        company_industry.append(industry)
        industry_topic = topics[industry,:]
        # p(word from background| document d)
        theta = beta.rvs(a,b)
        # how many words come from the background topic
        z = binom.rvs(words_p_company, theta)
        doc_background = multinomial(z, background_topic).rvs()
        doc_industry   = multinomial(words_p_company-z, industry_topic).rvs()
        
        companies[i,:] = (doc_background + doc_industry)
        Z.append(z)
    
    companies = csr_matrix(companies)
    
    return companies, Z, company_industry, topics, background_topic, word2id, id2word