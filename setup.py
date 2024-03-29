from distutils.core import setup
setup(
  name = 'connectome_to_model',         
  packages = ['connectome_to_model'],   
  version = '0.1.0',      
  license='GPL-3.0',        
  description = 'Toolbox for converting graphs into neural nets with feedback and biological inductive biases',
  author = 'Mashbayar Tugsbayar, Mingze Li',                   
  author_email = 'mashbayar@mila.quebec',     
  url = 'https://github.com/masht18/connectome-to-model.git',   
  download_url = 'https://github.com/masht18/connectome-to-model/archive/refs/tags/v0.1.1.tar.gz',
  keywords = ['CONNECTIVITY', 'TOP-DOWN', 'FEEDBACK'],  
  install_requires=[           
          'torch',
          'pandas',
          'soundfile',
          'gzip',
          'torchaudio',
          'numpy',
          'glob',
          'importlib',
          'matplotlib',
          'torchvision',      
      ],
  classifiers=[
    'Development Status :: 3 - Alpha', 
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GPL-3.0',   
    'Programming Language :: Python :: 3.11',
  ],
)
