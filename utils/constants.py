

#---------------------------------------PDF related constants-----------------------------
#removing top and bottom percentage from the pdf {as it contains supplier title and date and link}
top_margin = 0
bottom_margin = 0.1


#---------------------------------------Text Files constants---------------------------------
#path to pdf lists {pdf_id,pdf_url}
link_file_final = 'data\\links.csv'
#path to text files for each pdf ID.txt
result_text_path = 'data\\text_files\\'
#final csv file for training
debug_path = 'data\\debug_data_out.csv'

#---------------------------------------Models Parameters------------------------------------
#minimum letters allowed in pdf to be valid
min_Letters_in_text = 300

#number of features selected form the text
feature_number = 2000

#number of features selected after calculating importance from feature_number featuer
selected_features = 600

#number of estimators in the 
n_estimators = 400

#selected labels for the model any other label than those labels will me classified as others
#the following multiple values for the variable are some compinantions used in training comment if you want to remove, 
#last one is the only wprking
selected_labels = ['C of C_ROHS_Part,C of C_China_Part,C of C_REACH_Part,C of C_Prop65_Part','C of C_Korea_Generic','C of C_Korea_Part','C of C_REACH_Generic','C of C_REACH_Generic_Family','C of C_REACH_Generic_PKG','C of C_REACH_Generic_PL','C of C_REACH_Part','C of C_ROHS_Generic','C of C_ROHS_Generic,C of C_China_Generic','C of C_ROHS_Generic,C of C_China_Generic,C of C_Korea_Generic','C of C_ROHS_Generic,C of C_China_Generic,C of C_REACH_Generic','C of C_ROHS_Generic,C of C_China_Generic,C of C_WEEE_Generic','C of C_ROHS_Generic,C of C_ELV_Generic','C of C_ROHS_Generic,C of C_ELV_Generic,C of C_REACH_Generic','C of C_ROHS_Generic,C of C_ELV_Generic,C of C_REACH_Generic,C of C_Conflict_Minerals','C of C_ROHS_Generic,C of C_ELV_Generic,C of C_WEEE_Generic','C of C_ROHS_Generic,C of C_REACH_Generic','C of C_ROHS_Generic,C of C_REACH_Generic,C of C_Conflict_Minerals','C of C_ROHS_Generic,C of C_WEEE_Generic','C of C_ROHS_Generic,C of C_WEEE_Generic,C of C_ELV_Generic','C of C_ROHS_Generic,C of C_WEEE_Generic,C of C_ELV_Generic,C of C_REACH_Generic','C of C_ROHS_Generic,C of C_WEEE_Generic,C of C_REACH_Generic','C of C_ROHS_Generic,C of C_WEEE_Generic,C of C_REACH_Generic,C of C_Conflict_Minerals','C of C_ROHS_Part','C of C_ROHS_Part,C of C_China_Part','C of C_ROHS_Part,C of C_China_Part,C of C_ELV_Part','C of C_ROHS_Part,C of C_China_Part,C of C_ELV_Part,C of C_REACH_Part','C of C_ROHS_Part,C of C_China_Part,C of C_ELV_Part,C of C_WEEE_Part,C of C_REACH_Part','C of C_ROHS_Part,C of C_China_Part,C of C_Korea_Part','C of C_ROHS_Part,C of C_China_Part,C of C_REACH_Part','C of C_ROHS_Part,C of C_China_Part,C of C_REACH_Part,C of C_Japan_Part','C of C_ROHS_Part,C of C_China_Part,C of C_WEEE_Part,C of C_REACH_Part','C of C_ROHS_Part,C of C_Conflict_Minerals','C of C_ROHS_Part,C of C_ELV_Part','C of C_ROHS_Part,C of C_ELV_Part,C of C_REACH_Part','C of C_ROHS_Part,C of C_REACH_Generic','C of C_ROHS_Part,C of C_REACH_Part','C of C_ROHS_Part,C of C_REACH_Part,C of C_Conflict_Minerals','C of C_ROHS_Part,C of C_REACH_Part,Country of origin','C of C_ROHS_Part,C of C_WEEE_Part','C of C_ROHS_Part,C of C_WEEE_Part,C of C_Conflict_Minerals,Country of origin','C of C_ROHS_Part,C of C_WEEE_Part,C of C_ELV_Part','C of C_ROHS_Part,C of C_WEEE_Part,C of C_REACH_Part','C of C_TSCA_Generic','C of C_TSCA_Part','C of C_WEEE_Generic','C of C_WEEE_Generic,C of C_REACH_Generic','C of C_WEEE_Part','Calf_ROHS','China_ROHS','China_ROHS,WEEE','China_ROHS,WEEE,REACH','Datasheet','Declaration of Conformity_CE','Declaration of Conformity_CE_UK','Declaration of Conformity_Others','Declaration of Conformity_UK','ELV','Environmental_XML','Euro_ROHS','Euro_ROHS,China_ROHS','Euro_ROHS,China_ROHS,ELV,REACH','Euro_ROHS,China_ROHS,ELV,WEEE,REACH','Euro_ROHS,China_ROHS,REACH','Euro_ROHS,China_ROHS,WEEE','Euro_ROHS,Conflict Minerals_Statement','Euro_ROHS,ELV','Euro_ROHS,REACH','Euro_ROHS,REACH,Conflict Minerals_Statement','Euro_ROHS,WEEE','Euro_ROHS,WEEE,ELV,REACH','Euro_ROHS,WEEE,REACH','Euro_ROHS,WEEE,REACH,ELV','PCN','Material Declaration','Material Declaration_Class1','Material Declaration_Class2','Material Declaration_Class3','Material Declaration_Class4','Material Declaration_Class5','Material Declaration_Class6','MSDS','Pb-Free Roadmap','REACH','Taiwan_ROHS','Test_Report','Test_Report_REACH','Test_Report_ROHS','Test_Report_ROHS_REACH','TIN WHISKER TEST REPORT','WEEE','C of C_ODC_Generic','C of C_ODC_Part','C of C_PFOA_Generic','C of C_PFOA_Generic,C of C_PFOS_Generic','C of C_PFOA_Part,C of C_PFOS_Part','C of C_PFOS_Generic','C of C_POPs_Generic','C of C_POPs_Part','C of C_ROHS_Generic,C of C_REACH_Generic,C of C_Prop65_Generic','C of C_ROHS_Part,C of C_REACH_Part,C of C_Prop65_Part','C of C_ROHS_Part,C of C_REACH_Part,C of C_Prop65_Part,C of C_TSCA_Part','C of C_ROHS_Part,C of C_REACH_Part,C of C_TSCA_Part','C of C_VOCs_Generic','C of C_VOCs_Part','Extended Minerals_Template Report']
selected_labels = ['Scrubbing_Source','Datasheet','Selection Guide','Others','Manual and Guide','Product Brief','Package Brief','Environmental','Life_Cycle_Document','PCN']



selected_labels=['Datasheet' ,'Manual and Guide', 'Country of origin' ,'Qualification_Report',
 'Others' ,'Environmental', 'C of C_Management', 'Package Brief' ,'PCN',
 'Financial' ,'Selection Guide', 'Acquisition_Source' ,'Life_Cycle_Document','News']



selected_labels=['Datasheet','Environmental','Others','Manual and Guide']
#not selected labels any labels in the following will be removed entirly from the dataset, 
#last one is the only working
#reasons of removing are conflict and similarity with other important classes
#not_selected_Labels = ['Package Brief','Selection Guide','Manual and Guide' ]
not_selected_Labels=[]
class_weights = {}

environmental_types= ['Country of origin',
                      'WEEE',
                      'C of C_ELV_Part',
                      'C of C_WEEE_Part',	
                      'Test_Report_ROHS',
                      'Test_Report_REACH',
                      'REACH',
                      'Material Declaration_Class6',
                      'Material Declaration',
                      'Declaration of Conformity_UK',
                      'Declaration of Conformity_Others',
                      'Declaration of Conformity_CE_UK',
                      'Declaration of Conformity_CE',
                      'China_ROHS',
                      'C of C_WEEE_Generic',
                      'C of C_TSCA_Part',
                      'C of C_TSCA_Generic',
                      'C of C_REACH_Part',
                      'C of C_REACH_Generic',
                      'C of C_Prop65_Part',
                      'C of C_Prop65_Generic',
                      'C of C_PFAS_Part',
                      'C of C_PFAS_Generic',
                      'C of C_Others',
                      'C of C_Management',
                      'C of C_Conflict_Minerals',
                      'C of C_China_Part',
                      'C of C_China_Generic',
                      'C of C_Agency Approvals_Safety',
                      'C of C_Agency Approvals_Quality',
                      'C of C_ROHS_Part,C of C_REACH_Part',
                      'Euro_ROHS',
                      'C of C_ROHS_Part',
                      'C of C_ROHS_Generic',
                      'Others'
]                      
#assigning weight for different classes 
for i in selected_labels:
    
    class_weights[i]=1
class_weights['Datasheet'] = 30
class_weights['Manual and Guide'] = 5
class_weights['Environmental'] = 20
class_weights['Application Brief']=5
class_weights['Product Brief']=5
class_weights['Package Brief']=5
class_weights['Package Drawing']=5
multilevel_Classes = ['Environmental']

for type in environmental_types:
    class_weights[type]=10


td_idf_selected_words = ['accelerated' 'acceptance' 'accepted' 'accordance' 'accounting'
 'accurate' 'acid' 'acquisition' 'act' 'activation' 'active' 'adapter'
 'aecq' 'affected' 'agent' 'agreed' 'alternative' 'animal' 'annex'
 'annual' 'anthracene' 'applicable' 'application' 'approved' 'april'
 'article' 'assay' 'assembled' 'asset' 'assurance' 'attached' 'attachment'
 'audit' 'aug' 'authorization' 'autoclave' 'automotive' 'bacterial' 'bake'
 'batch' 'battery' 'bbp' 'believed' 'best' 'bias' 'billion' 'biological'
 'biphenyls' 'bipolar' 'bit' 'breathing' 'business' 'buy' 'bv' 'cable'
 'cadmium' 'calculated' 'calculation' 'camera' 'candidate' 'capacitance'
 'capital' 'care' 'cash' 'catalog' 'cdm' 'certain' 'certificate'
 'certification' 'certifies' 'certify' 'cfr' 'chain' 'change' 'changed'
 'channel' 'chapter' 'characteristic' 'charged' 'check' 'chemical' 'china'
 'chn' 'chromate' 'chromium' 'ci' 'class' 'click' 'clothing' 'coil'
 'collected' 'comment' 'commission' 'committee' 'common' 'communication'
 'company' 'companys' 'compliance' 'complies' 'comply' 'component'
 'compound' 'concentration' 'concern' 'condition' 'conduct' 'confidence'
 'conflict' 'conformity' 'consent' 'consequence' 'considered'
 'consolidated' 'contact' 'contain' 'contamination' 'contract' 'convey'
 'copyright' 'corning' 'corporate' 'council' 'country' 'creation' 'crh'
 'criterion' 'cs' 'current' 'custom' 'customer' 'cycle' 'cycling' 'data'
 'database' 'date' 'db' 'dbm' 'dbp' 'dc' 'dear' 'december' 'declaration'
 'declared' 'dehp' 'del' 'department' 'derated' 'described' 'description'
 'detection' 'determine' 'determined' 'development' 'device' 'dham'
 'dimension' 'direct' 'directive' 'director' 'discharge' 'disclosure'
 'discontinued' 'discrete' 'displayport' 'document' 'drawing' 'duration'
 'ec' 'echa' 'ed' 'effective' 'electrostatic' 'ema' 'email' 'employee'
 'en' 'end' 'ended' 'endotoxin' 'energy' 'eol' 'ep' 'equity' 'equivalent'
 'error' 'esd' 'established' 'estimate' 'ether' 'eu' 'european' 'ev'
 'evaluation' 'exceed' 'exchange' 'executive' 'exemption' 'exhibit'
 'expense' 'expert' 'expiration' 'explanation' 'exposure' 'expressly'
 'extinguishing' 'eye' 'fab' 'failure' 'feature' 'february' 'feed' 'feel'
 'filed' 'filer' 'financial' 'fiscal' 'fit' 'following' 'follows' 'form'
 'forwardlooking' 'france' 'free' 'frequency' 'ft' 'gandhi' 'ghz'
 'gigabit' 'gmbh' 'governance' 'government' 'grade' 'great' 'group'
 'growth' 'guidance' 'guide' 'hazard' 'hazardous' 'hbm' 'hdmi' 'health'
 'heat' 'hexavalent' 'hg' 'high' 'history' 'hour' 'hours' 'hp' 'hrs'
 'htrb' 'hub' 'human' 'humidity' 'hz' 'id' 'imply' 'income' 'incorporated'
 'indicate' 'info' 'information' 'inhalation' 'inline' 'input'
 'inspection' 'installation' 'instruction' 'insure' 'intellectual'
 'intended' 'intentionally' 'interface' 'intermittent' 'investment'
 'investor' 'iol' 'irritation' 'iso' 'issued' 'item' 'jan' 'jeita' 'jesda'
 'jose' 'js' 'jstd' 'jun' 'junction' 'june' 'key' 'khz' 'knowledge' 'kpa'
 'la' 'latest' 'lead' 'led' 'legal' 'length' 'letter' 'level' 'leviton'
 'liability' 'license' 'life' 'lifecycle' 'lifecycle_req' 'limit' 'list'
 'listed' 'load' 'lot' 'low' 'ma' 'management' 'manager' 'manual'
 'manufacture' 'manufacturer' 'manufacturing' 'mar' 'march' 'mark'
 'market' 'material' 'max' 'maximum' 'mean' 'measure' 'meeting' 'memory'
 'mercury' 'message' 'met' 'method' 'mgkg' 'mhz' 'micron' 'million'
 'milstd' 'min' 'mineral' 'mini' 'mixture' 'ml' 'mm' 'mode' 'moisture'
 'monitoring' 'monolithic' 'month' 'mounting' 'msl' 'mtbf' 'mtbfmttf'
 'mttf' 'murata' 'na' 'nd' 'negative' 'net' 'new' 'news' 'nexperia'
 'nexperiacom' 'nirav' 'nominal' 'notice' 'notification' 'nov' 'november'
 'number' 'nxp' 'obsolete' 'obtained' 'oct' 'officer' 'operated'
 'operating' 'option' 'order' 'origin' 'original' 'output' 'outside'
 'outstanding' 'owner' 'package' 'packaged' 'page' 'panasonic' 'pas'
 'patent' 'pbb' 'pbde' 'pci' 'pcn' 'performed' 'period' 'personnel' 'pf'
 'pg' 'phthalate' 'pin' 'plan' 'plant' 'pm' 'policy' 'polybrominated'
 'port' 'positive' 'power' 'powered' 'ppm' 'practice' 'pre'
 'preconditioning' 'presented' 'president' 'press' 'prior' 'process'
 'prodcompliancestartechcom' 'product' 'production' 'products' 'profit'
 'prohibited' 'property' 'protective' 'psia' 'publication' 'published'
 'publisher' 'pulse' 'pursuant' 'qfn' 'qualification' 'qualified'
 'quality' 'quantity' 'quarter' 'questions' 'quick' 'quotation'
 'radiation' 'range' 'rate' 'rated' 'rating' 'reach' 'reactivity' 'reason'
 'received' 'record' 'refer' 'reflow' 'regards' 'regional' 'registrant'
 'registration' 'regulation' 'reject' 'release' 'released' 'reliability'
 'reliable' 'replaced' 'replacement' 'report' 'reporting' 'representative'
 'reproduction' 'requirement' 'reserved' 'resin' 'resistance' 'restricted'
 'restriction' 'result' 'revenue' 'reverse' 'reviewed' 'revision' 'rh'
 'rights' 'risk' 'road' 'rohm' 'rohs' 'rule' 'sa' 'sale' 'sample' 'san'
 'sata' 'science' 'scope' 'search' 'section' 'security' 'seller'
 'semiconductor' 'sent' 'sep' 'series' 'sfp' 'share' 'shareholder' 'ship'
 'shipment' 'shown' 'silicon' 'siliconexpert' 'sirs' 'size' 'skin'
 'smelter' 'social' 'sodium' 'solderability' 'soldering' 'solution'
 'specification' 'specifications' 'st' 'standard' 'startechcom' 'stated'
 'statement' 'status' 'sterility' 'stock' 'stored' 'stress' 'subject'
 'substance' 'substances' 'summary' 'supplier' 'suppliers'
 'sustainability' 'svhc' 'systems' 'taiwan' 'tamb' 'tax' 'tc' 'technology'
 'tel' 'temperature' 'test' 'tested' 'testing' 'tests' 'th' 'thereof' 'ti'
 'time' 'tj' 'tjmax' 'tolerance' 'ton' 'transaction' 'transceiver' 'tstg'
 'typ' 'type' 'typical' 'update' 'usa' 'usb' 'usbc' 'use' 'usp' 'valid'
 'valve' 'vdc' 'version' 'vga' 'vi' 'vice' 'visual' 'voltage' 'vr'
 'vvc_mcp' 'wafer' 'water' 'website' 'weight' 'written' 'ww' 'xc_tyc'
 'year' 'years' 'yen' 'δtj']

#---------------------------------------OCR Engine Paths------------------------------------

poopler_Path = r'C:\poppler-24.02.0\Library\bin'
tesseract_path = r'C:\Users\161070\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

