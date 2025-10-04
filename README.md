### DESAFIO NASA SPACE APPS - EQUIPE COSMIC HACKERS
# Mateus Rosolem Baroni, Leandro Aguiar Mota, Luan Marques Batista, Otavio Augusto Teixeira, Yhan Pena
## Challenge: A World Away: Hunting for Exoplanets with AI

# Summary

    Data from several different space-based exoplanet surveying missions have enabled discovery of thousands of new planets outside our solar system, but most of these exoplanets were identified manually. With advances in artificial intelligence and machine learning (AI/ML), it is possible to automatically analyze large sets of data collected by these missions to identify exoplanets. Your challenge is to create an AI/ML model that is trained on one or more of the open-source exoplanet datasets offered by NASA and that can analyze new data to accurately identify exoplanets.

# Background

    Exoplanetary identification is becoming an increasingly popular area of astronomical exploration. Several survey missions have been launched with the primary objective of identifying exoplanets. Utilizing the “transit method” for exoplanet detection, scientists are able to detect a decrease in light when a planetary body passes between a star and the surveying satellite. Kepler is one of the more well-known transit-method satellites, and provided data for nearly a decade. Kepler was followed by its successor mission, K2, which utilized the same hardware and transit method, but maintained a different path for surveying. During both of these missions, much of the work to identify exoplanets was done manually by astrophysicists at NASA and research institutions that sponsored the missions. After the retirement of Kepler, the Transiting Exoplanet Survey Satellite (TESS), which has a similar mission of exoplanetary surveying, launched and has been collecting data since 2018.

    For each of these missions (Kepler, K2, and TESS), publicly available datasets exist that include data for all confirmed exoplanets, planetary candidates, and false positives obtained by the mission (see Resources tab). For each data point, these spreadsheets also include variables such as the orbital period, transit duration, planetary radius, and much more. As this data has become public, many individuals have researched methods to automatically identify exoplanets using machine learning. But despite the availability of new technology and previous research in automated classification of exoplanetary data, much of this exoplanetary transit data is still analyzed manually. Promising research studies have shown great results can be achieved when data is automatically analyzed to identify exoplanets. Much of the research has proven that preprocessing of data, as well as the choice of model, can result in high-accuracy identification. Utilizing the Kepler, K2, TESS, and other NASA-created, open-source datasets can help lead to discoveries of new exoplanets hiding in the data these satellites have provided.

# Objectives

    You may (but are not required to) consider the following:

    - Your project could be aimed at researchers wanting to classify new data or novices in the field who want to interact with exoplanet data and do not know where to start.
    - Your interface could enable your tool to ingest new data and train the models as it does so.
    - Your interface could show statistics about the accuracy of the current model.
    - Your model could allow hyperparameter tweaking from the interface.

# Resources
 
    - [Kepler Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
    - [K2 Planets and Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)
    - [TESS Objetcts of Interest](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)
    - [Exoplanet Detection Using Machine Learning](https://academic.oup.com/mnras/article/513/4/5505/6472249?login=false)
    - [Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification ](https://www.mdpi.com/2079-9292/13/19/3950)
    - [Near-Earth Object Surveillance Satellite Data](https://donnees-data.asc-csa.gc.ca/en/dataset/9ae3e718-8b6d-40b7-8aa4-858f00e84b30)