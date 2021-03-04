#!/usr/bin/env python
# coding: utf-8

# # Università degli Studi di Milano Bicocca  <br/> Master's Degree in Data Science
# 
# ## Final Project - Foundation of Computer Science (F9101Q001)

# **Gruppo 35**: <br/>
# Daniele Rizzo *872359*<br/>
# Carlo Salaorni *760080*<br/>
# Matteo Tarli *800676*
# 
# <a href = "https://github.com/gdv/foundationsCS/blob/master/progetti/2021-students.ipynb" title = 2021-students.ipynb>Assignment Body</a>

# Definizione delle librerie di interesse e i dataset di partenza.
# Le librerie `Pandas` e `NumPy` saranno utilizzate per la lettura ed elaborazione dei dataframe, mentre la libreria `re` per l'uso dell'espressioni regolari. Verrà infine sfruttata la libreria `time` per registrare il tempo di una run completa del codice.

# In[1]:


import time
start = time.time()


# In[2]:


import pandas as pd 
import re
import numpy as np


# In[3]:


times = pd.read_csv('https://github.com/DanieleRizz0/CSProject/raw/main/timesData.csv', thousands = ',')
shangai = pd.read_csv('https://github.com/DanieleRizz0/CSProject/raw/main/shanghaiData.csv')
cwur = pd.read_csv('https://github.com/DanieleRizz0/CSProject/raw/main/cwurData.csv')


# In[4]:


times.head()


# Si può notare che l'attributo `income` di interesse risulta essere di tipo `object`. Al fine di correggere questa situazione per utilizzi futuri di questa colonna si effettua una conversione in tipo `numeric`.

# In[5]:


times.income.dtype


# In[6]:


times.income = pd.to_numeric(times.income, errors = 'coerce')
times.head()


# Si noti che il record 4 presenta il valore `-` per l'attributo income. L'argomento `errors = 'coerce'` quindi viene utilizzato al fine di convertirlo in valore nullo.

# In[7]:


shangai.head()


# In[8]:


cwur.head()


# ### 1. For each university, extract from the times dataset the most recent and the least recent data, obtaining two separate dataframes.
# 

# L'obiettivo è ottenere due dataframe separati contenenti le informazioni più e meno recenti per ogni università contenuti nel dataframe `times`. Questo si può completare sfruttando i metodi di Pandas `idxmax` e `idxmin` a seguito di un raggruppamento rispetto al nome delle università. <br/>
# Questi metodi consentono di ottenere le posizioni dei record relativi rispettivamente al valore massimo e minimo. Una volta ottenuti questi indici è sufficiente sfruttare il metodo `loc` per estrarre l'informazione corrispondente. 

# In[9]:


new_times = times.loc[times.groupby('university_name').year.idxmax()]
new_times.head()


# In[10]:


old_times = times.loc[times.groupby('university_name').year.idxmin()]
old_times.head()


# ### 2. For each university, compute the improvement in income between the least recent and the most recent data points.
# 

# L'idea è di performare una join dei dataset costruiti nel punto precedente sul nome dell'università e di valutare da questa unione l'eventuale differenza di reddito. 

# In[11]:


merged = pd.merge(new_times, old_times, on = 'university_name', suffixes = ('_new', '_old'))
merged.head()


# E' importante individuare eventuali valori mancanti in queste colonne prima di effettuare il calcolo della variazione del reddito. Questa operazione si può effettuare con il comando `all`, il quale restituisce `False` se anche un solo valore della serie è nullo.

# In[12]:


merged.income_new.notna().all()


# In[13]:


merged.income_old.notna().all()


# In entrambi i dataframe sono presenti valori nulli. Al fine di risolvere questa problematica si opta per un'operazione di rimozione dei record con valori mancanti.

# In[14]:


merged.dropna(subset = ['income_new', 'income_old'], inplace = True)


# In[15]:


merged['improvement'] = merged.income_new - merged.income_old 


# In[16]:


merged[['university_name', 'improvement']].head()


# ### 3. Find the university with the largest increase computed in the previous point.
# 

# Per individuare l'università con la maggiore crescita si può sfruttare nuovamente il metodo `idxmax` con riferimento alla colonna `improvement`. E' doveroso però effettuare un controllo per evitare l'evenienza in cui più università presentino il medesimo miglioramento. Per questa operazione si effettua un ordinamento in base al miglioramento e si controllano eventuali valori massimi ripetuti.

# In[17]:


merged.improvement.sort_values(ascending = False)


# L'università *TU Dresden* risulta essere quella che ha registrato un maggiore incremento nel livello di reddito nel periodo di analisi (pari a 67.8).

# In[18]:


merged.loc[merged['improvement'].idxmax()]


# ### 4. For each ranking, consider only the most recent data point. For each university, compute the maximum difference between the rankings (e.g. for Aarhus University the value is 122-73=49). Notice that some rankings are expressed as a range.

# Nei ranking sono presenti alcuni problemi di eterogeneità relativi al nome delle università. Per esempio nel ranking `shangai` l'università *Massachusetts Institute of Technology* viene indentificata anche con la nota sigla *MIT* tra parentesi. Per questo motivo, prima di operare al fine di dare una risposta al quesito, è necessario normalizzare i nomi delle università nei diversi ranking.
# Le operazioni che svolgeremo saranno volte a rendere tutte le stringhe minuscole e ad eliminare eventuali parentesi.
# 
# Infine, nel dataframe `times`, alcuni record hanno un valore della posizione preceduta da un segno `=` che va necessariamente eliminato per poter trattare tutte le posizioni come valori numerici.
# 
# Per svolgere queste operazioni sfrutteremo sia le espressioni regolari fornite dalla libreria `re`, sia i metodi associati alle stringhe.

# Per prima cosa si procedere alla compilazione dell'espressione regolare che verrà usata per gestire gli intervalli di valori.

# In[19]:


src = re.compile('(?P<first>\d+)\s*-*\s*(?P<second>\d*)')


# ###### `times`

# In[20]:


times.university_name = times.university_name.str.lower()
times.university_name.head()


# In[21]:


times.world_rank = times.world_rank.str.strip('=')
times.world_rank


# ###### `shangai`

# In[22]:


shangai.university_name = shangai.university_name.str.lower()
shangai.university_name.head()


# In[23]:


shangai.university_name = shangai.university_name.str.replace('\(\w+\)$', '', regex = True)
shangai.university_name.head()


# ###### `cwur`

# In[24]:


cwur.institution = cwur.institution.str.lower()
cwur.institution.head()


# In[25]:


cwur.institution = cwur.institution.str.replace('\(\w+\)$', '', regex = True)
cwur.institution.head()


# A questo punto è possibile proseguire con le operazioni. Per prima cosa individuiamo l'anno più recente contenuto nei diversi ranking ed andiamo ad integrarli ai precedenti dataset considerando come chiave il nome dell'università.

# In[26]:


times_years = set(times.year)
times_years


# In[27]:


shangai_years = set(shangai.year)
shangai_years


# In[28]:


cwur_years = set(cwur.year)
cwur_years


# In[29]:


rank = pd.merge(times[times.year == max(times_years)].add_suffix('_times'), shangai[shangai.year == max(shangai_years)].add_suffix('_shangai'), left_on = 'university_name_times', right_on = 'university_name_shangai')
rank = rank.merge(cwur[cwur.year == max(cwur_years)].add_suffix('_cwur'), right_on = 'institution_cwur', left_on = 'university_name_times')
rank.rename({'university_name_times':'university_name'}, axis = 'columns', inplace = True)


# E' necessario osservare che nei dataset `shangai` e `times` il ranking è espresso non soltanto con valori puntuali ma anche per mezzo di intervalli. Quest'ultimi verranno elaborati in modo tale da ottenere il valore centrale degli stessi, così da poter verificare infine la massima differenza tra le varie posizioni.

# In[30]:


for i in range(len(rank)):
    fnd = src.search(rank.loc[i,'world_rank_shangai'])
    if fnd.group('second'): 
        rank.loc[i, 'world_rank_shangai_2'] = (int(fnd.group('first')) + int(fnd.group('second')))/2
    else:
         rank.loc[i, 'world_rank_shangai_2'] = int(fnd.group('first'))


# In[31]:


for i in range(len(rank)):
    fnd = src.search(rank.loc[i,'world_rank_times'])
    if fnd.group('second'): 
        rank.loc[i, 'world_rank_times_2'] = (int(fnd.group('first')) + int(fnd.group('second')))/2
    else:
         rank.loc[i, 'world_rank_times_2'] = int(fnd.group('first'))


# In[32]:


rank.columns


# Il dataframe così costruito presenta un'ampia complessità dovuta alla grande quantità di colonne. Ma solo tre di queste sono di nostro interesse: `world_rank_times_2`,`world_rank_shangai_2`, `world_rank_cwur`. Si noti che per distinguere il posizionamento espresso come intervallo da quello puntuale, per i primi due ranking, è stato aggiunto il suffisso `_2`.

# In[33]:


rank[['world_rank_times_2','world_rank_shangai_2', 'world_rank_cwur']]


# Con le informazioni contenute del dataframe precedente è possibile computare le differenze di classifica e selezionare così quella massima tra i diversi ranking.

# In[34]:


for i in range(len(rank)):
    rank.loc[i,'max_diff'] = max(abs(rank.loc[i,'world_rank_times_2'] - rank.loc[i,'world_rank_shangai_2']), abs(rank.loc[i,'world_rank_times_2'] - rank.loc[i,'world_rank_cwur']), abs(rank.loc[i,'world_rank_shangai_2'] - rank.loc[i,'world_rank_cwur']))


# In[35]:


rank.sort_values('max_diff', ascending = False)[['university_name','world_rank_times_2', 'world_rank_shangai_2','world_rank_cwur', 'max_diff']]


# Per validare il risultato dell'operazione effettuiamo un confronto col valore dell'università `Aarhus university` (max_diff = **49**).

# In[36]:


rank[rank.university_name == 'aarhus university'].max_diff


# ### 5. Consider only the most recent data point of the times dataset. Compute the number of male and female students for each country.
# 

# Per prima cosa estraiamo i dati più recenti dal dataframe `times`. Si effettua quindi una copia del dataframe in modo da operare in maggiore libertà sul subset. In questo caso non è un problema dato che la dimensione del dataframe originale non è tale da rendere la copia svantaggiosa.

# In[37]:


recent_times = times[times.year == max(times_years)].copy()


# In[38]:


recent_times.female_male_ratio.isnull().any()


# Essendo presente un valore nullo si opta anche in questo caso per una rimozione di tale valore. Si effettua anche un controllo sui rapporti definiti come stringhe per verificare siano tutti validi.

# In[39]:


recent_times.dropna(subset = ['female_male_ratio'], inplace = True)


# In[40]:


recent_times = recent_times[recent_times['female_male_ratio'].str.contains('\d+\s*:\s*\d+')]


# In[41]:


recent_times.head()


# In[42]:


recent_times['num_students'].dtype


# Si effettua ora una conversione numerica del rapporto $\frac{femmina}{maschio}$ a partire dalla colonna di stringhe e successivamente si calcolano il numero di maschi e femmine sfruttando il numero di studenti totale.

# In[43]:


recent_times[['female', 'male']] = round((recent_times['female_male_ratio'].str.split(' : ', expand = True).astype('int')/100).mul(recent_times['num_students'], axis = 0))


# In[44]:


recent_times[['female', 'male']].sort_values('male')


# A questo punto è sufficiente raggruppare per la nazione e sommare il numero di studenti maschili e femminili per ognuna di queste, ottenendo quindi il nostro risultato.

# In[45]:


recent_times[['country','female', 'male']].groupby('country').sum()


# ### 6. Find the universities where the ratio between female and male is below the average ratio (computed over all universities).

# In[46]:


recent_times[recent_times.male == 0]


# In[47]:


recent_times[recent_times['female_male_ratio'].isna()]


# Non ci sono valori nulli per il rapporto ma c'è un problema legato all'università femminile `ewha womans university` che fornisce un rapporto $\frac{female}{male}$ pari a $\infty$ e di conseguenza considerare questa università nel calcolo della media porterebbe questa a divergere. 
# Per questo motivo la `ewha womans university` viene esclusa e considerata a priori superiore alla media.

# In[48]:


recent_times_under = recent_times[recent_times['male'] != 0].copy()


# In[49]:


recent_times_under['proper_ratio'] = recent_times['female'] / recent_times['male']
average_ratio = recent_times_under['proper_ratio'].mean() #media geometrica
recent_times_under = recent_times_under[recent_times_under['proper_ratio'] < average_ratio]


# In[50]:


average_ratio


# In[51]:


recent_times_under[['world_rank', 'university_name', 'proper_ratio']]


# ### 7. For each country compute the fraction of students that are in one of the universities computed in the previous point (that is, the denominator of the ratio is the total number of students over all universities in the country).

# Si creano due serie, una contenente il numero di studenti per nazione ed una per università. Si uniscono i due sulla base della nazione in modo da poter calcolare infine i rapporti, facendo leva sulle funzionalità di Pandas.

# In[52]:


student_university = recent_times_under.groupby(['university_name', 'country'])['num_students'].sum()
student_university


# In[53]:


student_country = recent_times.groupby('country')['num_students'].sum()


# In[54]:


student_merged = pd.merge( student_country, student_university, right_on = 'country', left_index = True, suffixes = ('_country', '_university') )
student_merged.reset_index(inplace = True)
student_merged


# In[55]:


student_merged['ratio'] = student_merged['num_students_university']/student_merged['num_students_country']
student_merged


# ### 8. Read the file educational_attainment_supplementary_data.csv, discarding any row without country_name or series_name.

# Per questa richiesta abbiamo usato il metodo `dropna` con flag `subset` per eliminare le righe con `null` nelle due colonne indicate. Si verifica la riduzione del numero di righe con il metodo `shape`.

# In[56]:


edu_att_supp_data = pd.read_csv('https://github.com/DanieleRizz0/CSProject/blob/main/educational_attainment_supplementary_data.csv?raw=true')


# In[57]:


edu_att_supp_data.shape


# In[58]:


edu_att_supp_data.dropna(subset = ['country_name', 'series_name'], inplace = True)
edu_att_supp_data


# In[59]:


edu_att_supp_data.shape


# ### 9. From attainment build a dataframe with the same data, but with 4 columns: country_name, series_name, year, value.

# Per risolvere il quesito si utilizzato il metodo di Pandas `stack`, che consente di trasporre le colonne del dataframe rappresentative degli anni, passando così da un dataset di tipo key figure-based ad uno account-based. Per non perdere i valori nulli nel calcolo, viene settato il flag `dropna` del metodo su `False`.

# In[60]:


edu_att_supp_data.head(1)


# In[61]:


four_columns_att = edu_att_supp_data.set_index(['country_name', 'series_name'])
four_columns_att = four_columns_att.stack(dropna = False).reset_index()
four_columns_att.rename(columns = {'level_2': 'year', 0: 'value'}, inplace = True)
four_columns_att


# ### 10. For each university, find the number of rankings in which they appear (it suffices to appear in one year for each ranking).

# Per questa richiesta si sfruttano le operazioni di insieme. Si inseriscono le università presenti nei diversi ranking in altrettanti set. Infine si uniscono e si effettua un conteggio per il numero di volte che un'università si trova all'interno dell'insieme complessivo.

# In[62]:


uni_times = set(times['university_name'])
uni_shangai = set(shangai['university_name'])
uni_cwur = set(cwur['institution'])


# In[63]:


universities = pd.DataFrame(list(uni_times) + list(uni_shangai) + list(uni_cwur), columns = ['university'])


# In[64]:


universities.groupby('university').size()


# ### 11. In the times ranking, compute the number of times each university appears.

# Per questo punto si raggruppa il dataframe in base all'università utilizzando la funzione `groupby` di Pandas e si effettua un conteggio.

# In[65]:


times.groupby('university_name').size()


# ### 12. Find the universities that appear at most twice in the times ranking.
# 

# Si definisce una maschera booleana tale da ritornare `True` se le università sono presenti al più due volte nel ranking `times`.
# Infine si sfrutta questa maschera per estrarre le università corrispondenti.

# In[66]:


tf = times.groupby('university_name').size() <= 2
tf


# In[67]:


set(tf[tf == True].index)


# ### 13. The universities that, in any year, have the same position in all three rankings (they must have the same position in a year).

# Si costruisce un dataframe `full_rank` andando in join dei dataset `times`, `shangai` e `cwur` con riferimento non solo al nome dell'università ma anche con l'anno di riferimento del ranking. 
# Fatto ciò, si ricerca l'osservazione tale ha fatto registrare la stessa posizione nei diversi ranking e, per costruzione del dataframe `full_rank`, anche nello stesso anno.

# In[68]:


cwur.rename(columns = {'institution' : 'university_name'}, inplace = True)
cwur


# In[69]:


full_rank = times.merge(cwur, on = ['university_name','year'], how = 'inner' )
full_rank = full_rank.merge( shangai, on = ['university_name','year'], how = 'inner' )
full_rank.rename( columns = {'world_rank_x':'times_rank','world_rank_y':'cwur_rank','world_rank':'shangai_rank'}, inplace=True )


# In[70]:


full_rank['cwur_rank'] = full_rank['cwur_rank'].astype(str)


# In[71]:


tmp_merged = full_rank[(full_rank['times_rank'] == full_rank['cwur_rank']) & (full_rank['cwur_rank'] == full_rank['shangai_rank'])]


# In[72]:


tmp_merged[['university_name','year','times_rank','cwur_rank','shangai_rank']]


# In[73]:


end = time.time()

print("Il processo complessivo è stato ultimato in {} secondi.".format(int((end-start))))


# In[ ]:




