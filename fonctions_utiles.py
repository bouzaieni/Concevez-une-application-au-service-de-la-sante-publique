# Calcul du taux de remplissage  par colonne
def taux_remplissage_par_colonne(data):
    plt.figure(figsize=(15, 5))
    G = gridspec.GridSpec(1, 1)

    ax = plt.subplot(G[0, :])
    taux_remplissage = 100-data.isna().mean()*100
    ax = taux_remplissage.plot(kind='bar', color='red')
    ax.set_title('Taux de remplissage par colonne')
    ax.set_xlabel('Colonne')
    ax.set_ylabel('Taux de remplissage')
    ax.grid(True)
    fichier ='taux_remplissage'+'.png'
    plt.savefig(fichier)
    plt.show()
    
def imputation_mediane(df,col):
  df[col].fillna(df.groupby('main_category_fr')[col].transform('median'), inplace = True)
  # Il reste quelques valeurs nulles dont le groupement n'a pa de mediane. On les remplace par 0
  df[col].fillna(0, inplace = True)

def distribution_avant_apres(df_avant, df_apres, colonne_cible):
  colonne_cible_apres_imputation = colonne_cible+'_apres_imputation'
  sns.kdeplot(colonne_cible, data = df_avant, label=colonne_cible)
  sns.kdeplot(colonne_cible, data = df_apres, label=colonne_cible_apres_imputation)
  plt.title(f'comparaison des distributions prediction colonne {colonne_cible}')
  plt.legend()
  fichier = 'ancien_nouveau_'+colonne_cible+'.png'
  plt.savefig(fichier)
  plt.show()

def imputation_knn(df, colonne_cible):
  # selectionner les lignes ou la colonne_cible n'est pas nulle
  data_cible_index= df[df[colonne_cible].notna()].index
  # supprimer toutes les valeurs manquantes
  df_cible_index = df.loc[data_cible_index,colonnes_knn]
  df_cible_index = df_cible_index.dropna()
  data_colonne_cible = df_cible_index[colonne_cible]
  # supprimer la colonne cible
  del df_cible_index[colonne_cible]
  xtrain, xtest, ytrain, ytest = train_test_split(df_cible_index, data_colonne_cible, train_size=0.8)
  knn = neighbors.KNeighborsRegressor(n_neighbors=6)
  knn.fit(xtrain, ytrain)
  # Score R^2
  #print('\tscore R^2 : ',knn.score(xtest, ytest))
  # prédiction des valeurs manquantes pour la colonne_cible
  #df_apres_imputation = pd.DataFrame(columns=colonne_cible)
  df_apres_imputation=knn.predict(df[df_cible_index.columns.to_list()].fillna(value=0))
  return df_apres_imputation
  
def func_reverse_minmax(x):
    return -(x-40)-15

def analyse_univariee(data,colonne,label):
    print(f'moyenne : {round(data[colonne].mean(),2)}')
    print(f'mediane : {round(data[colonne].median(),2)}')
    print(f'mode : {round(data[colonne].mode(),2)}')
    print(f'variance : {round(data[colonne].var(),2)}')
    print(f'skewness : {round(data[colonne].skew(),2)}')
    print(f'kurtosis : {round(data[colonne].kurtosis(),2)}')
    print(f'ecart type : {round(data[colonne].std(),2)}')
    print(f'min : {round(data[colonne].min(),2)}')
    print(f'25% : {round(data[colonne].quantile(0.25),2)}')
    print(f'50% : {round(data[colonne].quantile(0.5),2)}')
    print(f'75% : {round(data[colonne].quantile(0.75),2)}')
    print(f'max : {round(data[colonne].max(),2)}')
    print(colored('Interprétation', 'red', attrs=['bold']))
    if np.floor(data[colonne].skew())==0:
        print('la distribution de la colonne '+colonne +' est symétrique')
    elif round(data[colonne].skew(),2)>0:
        print('la distribution de la colonne '+colonne + ' est étalée à droite')
    else:
        print('la distribution de la colonne '+colonne +' est étalée à gauche')
    
    if np.floor(data[colonne].kurtosis())==0:
        print('la distribution de la colonne '+colonne +' a le même aplatissement que la distribution normale')
    elif round(data[colonne].kurtosis(),2)>0:
        print('la distribution de la colonne '+colonne + ' est moins aplatie que la distribution normale')
    else:
        print('la distribution de la colonne '+colonne +' est plus aplatie que la distribution normale')
                   
    plt.figure(figsize=(15, 5))
    plt.subplot( 1,2 ,1)
    sns.boxplot(data[colonne], width=0.5, color='red')
    plt.title('Boite a moustache de la colonne '+label,fontsize=15)
    plt.subplot(1,2,2 )
    sns.histplot(data[colonne], kde=True, color='blue')
    plt.title('histogramme de la colonne  '+label,fontsize=15)        
    plt.show()       
    plt.tight_layout()
                 
                 
def camembert_fig(data, colonne, nb_val):
    lab = data[colonne].value_counts().index.values[0:nb_val]
    val = data_france['main_category_fr'].value_counts().values[0:nb_val]

    fig1, ax1 = plt.subplots()
    plt.figure(facecolor=None)
    ax1.pie(val, labels=lab, autopct='%1.1f%%',shadow=True, startangle=90)
    ax1.axis('equal')  
    ax1.set_title('Liste des valeurs',fontsize=15,c='red',fontstyle='italic',ha='center')
    fichier ='camembert_'+colonne+'.png'
    plt.savefig(fichier)
    plt.show()
    
    
def wordcloud_fig (data,colonne):
    text = " ".join(i for i in data[colonne])
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.figure( figsize=(7,12))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    fichier ='wordcloud_'+colonne+'.png'
    plt.savefig(fichier)
    plt.show()
    
# Test d'inependence Chi-square 
def test_chi2(cross_table):
  test_statistic, p_values, degrees_freedom, expected = chi2_contingency(cross_table) 
  # Résultats du test
  print('test statistic :',test_statistic)
  print('p_values du test :',p_values)
  print('degrees of freedom :',degrees_freedom)
  if p_values <0.05:
    print ('Il y a une corrélation entre les deux variables')

def cercle_correlation(pca, features, x, y):
    fig, ax = plt.subplots(figsize=(10, 9))
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0,
                 0,  # Start the arrow at the origin
                 pca.components_[x, i],  #0 for PC1
                 pca.components_[y, i],  #1 for PC2
                 head_width=0.07,
                 head_length=0.07, 
                 width=0.02,              )

        plt.text(pca.components_[x, i] + 0.05,
                 pca.components_[y, i] + 0.05,
                 features[i])

    # affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')


    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))


    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    fichier ="cercle_corrélation_F{}_F{})".format(x+1, y+1)+'.png'
    plt.savefig(fichier)
    plt.show(block=False)
    

def  recommandation(groupe_produit):
    # Afficher les score des 5 meilleurs produits recommandés

    plt.figure(figsize=(10, 5))
    pff =data_france[data_france['main_category_fr'] == groupe_produit][['image_url','product_name','score_ecolo_healthy']]
    index_healthy = pff['score_ecolo_healthy'].sort_values(ascending=True)[:5].index.to_list()
    pff = pd.DataFrame(pff.loc[index_healthy,'score_ecolo_healthy'].to_list(), index = pff.loc[index_healthy,'product_name'].to_list())
    
    pff.plot.bar(legend=None)
    plt.title('Les 5 meilleurs produits recommandés',fontsize=15)
    fichier ='score_meilleurs_5produits'+'.png'
    plt.savefig(fichier)
    plt.show()
    

    # Afficher les noms et images des 5 produits recommandés
    for i in index_healthy:
        url_image = data_france.loc[i,'image_url']
        display(data_france.loc[i,'product_name'])
        if url_image == 'image de produit non disponible':
          print(url_image)
        else:
          display(Image(url= url_image, width = 100, height = 120))
        print(150*'*')
        
