import { TestBed } from '@angular/core/testing';

import { FuzzyService } from './fuzzy.service';

describe('FuzzyService', () => {
  let service: FuzzyService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(FuzzyService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
