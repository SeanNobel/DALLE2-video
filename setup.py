from setuptools import setup, find_packages

# exec(open("dalle2_video/version.py").read())

setup(
    name="dalle2-video",
    packages=find_packages(exclude=[]),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "dalle2_video = dalle2_video.cli:main",
            # 'dream = dalle2_pytorch.cli:dream'
        ],
    },
    version="0.0.1",
    # license="MIT",
    description="Video generation based from CLIP embeddings.",
    author="Sensho Nobe",
    author_email="sean.y.nobel@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/SeanNobel/DALLE2-video",
    keywords=["artificial intelligence", "deep learning", "text to image"],
    install_requires=[
        "dalle2-pytorch",
    ],
    #   classifiers=[
    #     'Development Status :: 4 - Beta',
    #     'Intended Audience :: Developers',
    #     'Topic :: Scientific/Engineering :: Artificial Intelligence',
    #     'License :: OSI Approved :: MIT License',
    #     'Programming Language :: Python :: 3.6',
    #   ],
)
