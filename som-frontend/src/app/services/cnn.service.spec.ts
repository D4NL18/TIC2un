import { TestBed } from '@angular/core/testing';

import { CnnService } from './cnn.service';

describe('CnnService', () => {
  let service: CnnService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(CnnService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
