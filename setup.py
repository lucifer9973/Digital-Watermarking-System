from setuptools import setup, find_packages

setup(
    name="digital_watermark",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask==2.3.3',
        'numpy<2',
        'opencv-python==4.8.1.78',
        'PyWavelets==1.4.1',
        'scikit-image==0.22.0',
    ]
)