# SoundFinder

SoundFinder is an intelligent audio sample retrieval tool designed to integrate with a music production workflow using FL Studio.

Given a reference audio track, SoundFinder analyzes and separates its audio components, identifies the corresponding instruments using machine learning, and searches the user's local sample library for acoustically similar sounds.

## Audio input specification

The audio loader accepts WAV files and 320 kb/s MP3 files. Files longer than 4 seconds are rejected. Accepted files are converted to mono, resampled to 44.1 kHz, and represented internally as `float32` samples.

## Project status

SoundFinder is currently under development as part of a student project.

## V1 Objectives

- Import and analyze a reference audio track
- Separate the track into exploitable audio components
- Identify instruments using machine learning
- Index a local sample library
- Search for acoustically similar samples
- Return and preview the most relevant samples
- Integrate with the FL Studio workflow

## Team

Student project developed by a team of 5.

## Charles 9/11/2026
Le fichier loader.py contient la logique d’acceptation des fichiers audio. SoundFinder accepte les fichiers WAV ainsi que les MP3 à 320 kb/s, refuse les autres extensions, les fichiers inexistants, les fichiers de plus de 4 secondes et les MP3 dont le débit est différent de 320 kb/s. Les fichiers acceptés sont convertis en mono, rééchantillonnés à 44,1 kHz et transformés en float32.

Les tests de test_audio_loader.py vérifient la normalisation d’un WAV stéréo, le rééchantillonnage, le format float32, le rejet d’un fichier de plus de 4 secondes et le rejet d’une extension non supportée. Au total, les quatre tests du projet passent avec succès.