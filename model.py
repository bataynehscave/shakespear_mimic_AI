# model = Sequential()
# model.add(Input((SEQ_LENGTH, len(characters))))
# model.add(LSTM(128))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# model.fit(x,y, batch_size=256, epochs=4)

# model.save('shake_spear_gen_1.keras')